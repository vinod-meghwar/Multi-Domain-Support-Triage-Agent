import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from retriever import RetrievedDoc, SupportRetriever
from utils import TriageResult


LOGGER = logging.getLogger(__name__)

# ✅ FIX: Removed "access" and "security" as standalone triggers — they caused
# almost every ticket to escalate. Now only multi-word or specific terms escalate.
SENSITIVE_KEYWORDS = [
    "billing",
    "charge",
    "charged",
    "refund",
    "fraud",
    "stolen",
    "unauthorized",
    "hacked",
    "security breach",
    "account recovery",
    "locked out",
    "admin access",
    "password reset",
    "identity verification",
    "unauthorized charge",
    "phishing",
    "scam",
]

# ✅ Single-word sensitive keywords (only escalate if standalone context matches)
SENSITIVE_SINGLE_WORDS = {
    "fraud",
    "stolen",
    "unauthorized",
    "hacked",
    "phishing",
    "scam",
    "refund",
    "billing",
}

REQUEST_TYPE_RULES = {
    "billing_issue": ["billing", "charge", "charged", "invoice", "refund", "payment"],
    "fraud_report": ["fraud", "unauthorized", "stolen", "scam", "phishing"],
    "account_access": ["locked out", "account recovery", "password reset", "login issue"],
    "permissions": ["permission", "permissions", "role", "admin access", "workspace access"],
    "security_concern": ["security breach", "vulnerability", "hacked", "breach"],
    "bug_report": ["bug", "error", "fails", "failure", "broken", "crash", "not loading", "cannot load"],
    "feature_request": ["feature", "enhancement", "improve", "add support", "request for"],
}


@dataclass
class TriageDecision:
    request_type: str
    product_area: str
    decision: str
    justification: str
    response: str
    confidence: float


class SupportTriageAgent:
    def __init__(self, retriever: SupportRetriever, min_confidence: float = 0.15) -> None:
        self.retriever = retriever
        self.min_confidence = min_confidence

    def triage(self, ticket: str) -> TriageResult:
        request_type = self._identify_request_type(ticket)
        retrieved = self.retriever.search(ticket, top_k=3)
        confidence = retrieved[0].score if retrieved else 0.0
        product_area = self._classify_product_area(ticket, retrieved)

        sensitive = self._contains_sensitive_terms(ticket)
        unsupported = not retrieved or confidence < self.min_confidence

        if sensitive:
            decision = "escalate"
            justification = (
                "Escalated due to sensitive policy category "
                "(billing/fraud/account access/permissions/security)."
            )
            response = (
                "This request involves a sensitive topic that requires secure human review. "
                "I have escalated it to a support specialist who will assist you shortly."
            )
        elif unsupported:
            decision = "escalate"
            justification = (
                "Escalated because no sufficiently relevant documentation was retrieved "
                f"(top confidence={confidence:.2f}, threshold={self.min_confidence})."
            )
            response = (
                "I could not find a verified answer in the support documentation for your issue. "
                "This has been escalated to a human support agent for assistance."
            )
        else:
            decision = "respond"
            response = self._grounded_response(ticket, retrieved)
            sources = ", ".join(f"{doc.source} ({doc.score:.2f})" for doc in retrieved[:2])
            justification = f"Response grounded in retrieved docs: {sources}."

        llm_result = self._llm_json_wrapper(
            request_type=request_type,
            product_area=product_area,
            decision=decision,
            justification=justification,
            response=response,
            confidence=confidence,
        )

        return TriageResult(
            request_type=llm_result["request_type"],
            product_area=llm_result["product_area"],
            decision=llm_result["decision"],
            justification=llm_result["justification"],
            response=llm_result["response"],
        )

    def _identify_request_type(self, ticket: str) -> str:
        lowered = ticket.lower()
        for request_type, keywords in REQUEST_TYPE_RULES.items():
            if any(keyword in lowered for keyword in keywords):
                return request_type
        return "general_support"

    def _classify_product_area(self, ticket: str, retrieved: List[RetrievedDoc]) -> str:
        lowered = ticket.lower()
        if "hackerrank" in lowered or "coding test" in lowered or "challenge" in lowered or "compiler" in lowered:
            return "HackerRank Support"
        if "claude" in lowered or "anthropic" in lowered or "workspace" in lowered:
            return "Claude Help Center"
        if "visa" in lowered or "card" in lowered or "transaction" in lowered or "merchant" in lowered:
            return "Visa Support"
        if retrieved:
            return retrieved[0].domain
        return "Unknown"

    def _contains_sensitive_terms(self, ticket: str) -> bool:
        lowered = ticket.lower()
        # Check multi-word sensitive phrases first (more specific)
        for phrase in SENSITIVE_KEYWORDS:
            if " " in phrase and phrase in lowered:
                return True
        # Check single-word sensitive terms
        words = set(re.findall(r"[a-zA-Z0-9_]+", lowered))
        return bool(words.intersection(SENSITIVE_SINGLE_WORDS))

    def _grounded_response(self, ticket: str, retrieved: List[RetrievedDoc]) -> str:
        snippets: List[str] = []
        query_terms = set(re.findall(r"[a-zA-Z0-9_]+", ticket.lower()))

        # Remove very common words from query terms to avoid noise
        stopwords = {"i", "my", "the", "a", "an", "is", "it", "to", "and", "or", "in", "of", "for", "on", "with", "can", "not", "be", "this", "that", "have"}
        query_terms -= stopwords

        for doc in retrieved:
            clean_text = re.sub(r"^\s*#+\s*", "", doc.text, flags=re.MULTILINE)
            sentences = re.split(r"(?<=[.!?])\s+", clean_text)
            for sentence in sentences:
                normalized = re.sub(r"\s+", " ", sentence).strip()
                if len(normalized) < 30:
                    continue
                sentence_terms = set(re.findall(r"[a-zA-Z0-9_]+", normalized.lower()))
                sentence_terms -= stopwords
                overlap = len(query_terms.intersection(sentence_terms))
                if overlap >= 2:
                    snippets.append(normalized)
                if len(snippets) >= 3:
                    break
            if len(snippets) >= 3:
                break

        if not snippets:
            return (
                "Based on the support documentation, please follow the documented troubleshooting "
                "steps for your issue. If the problem persists, contact support for further help."
            )
        return re.sub(r"\s+", " ", " ".join(snippets[:2])).strip()

    def _llm_json_wrapper(
        self,
        request_type: str,
        product_area: str,
        decision: str,
        justification: str,
        response: str,
        confidence: float,
    ) -> dict:
        payload = {
            "request_type": request_type,
            "product_area": product_area,
            "decision": decision,
            "justification": f"{justification} Confidence={confidence:.2f}.",
            "response": response,
        }
        return json.loads(json.dumps(payload))
