import argparse
import logging
from pathlib import Path
from typing import List

from agent import SupportTriageAgent
from retriever import SupportRetriever
from utils import TriageResult, ensure_project_dirs, load_tickets, setup_logging, write_results


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root path",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "support_tickets.csv",
        help="Input tickets CSV path (relative to project root by default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "output.csv",
        help="Output CSV path (relative to project root by default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, candidate: Path) -> Path:
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


def run() -> Path:
    args = parse_args()
    project_root = args.project_root.resolve()

    # ✅ FIX: log.txt is written to project root (required for submission)
    log_file = project_root / "log.txt"
    setup_logging(args.log_level, log_file=log_file)

    ensure_project_dirs(project_root)

    input_csv = resolve_path(project_root, args.input)
    output_csv = resolve_path(project_root, args.output)
    docs_dir = project_root / "docs"

    LOGGER.info("=== Multi-Domain Support Triage Agent Starting ===")
    LOGGER.info("Project root: %s", project_root)
    LOGGER.info("Input CSV:    %s", input_csv)
    LOGGER.info("Output CSV:   %s", output_csv)
    LOGGER.info("Docs dir:     %s", docs_dir)
    LOGGER.info("Log file:     %s", log_file)

    retriever = SupportRetriever(docs_dir=docs_dir)
    agent = SupportTriageAgent(retriever=retriever)

    tickets = load_tickets(input_csv)
    LOGGER.info("Loaded %d ticket(s) from %s", len(tickets), input_csv)

    results: List[TriageResult] = []
    for idx, ticket in enumerate(tickets, start=1):
        result = agent.triage(ticket)
        results.append(result)
        LOGGER.info(
            "[%d/%d] %s -> %s (%s)",
            idx,
            len(tickets),
            result.request_type,
            result.decision,
            result.product_area,
        )
        LOGGER.debug("  Ticket:        %s", ticket[:120])
        LOGGER.debug("  Justification: %s", result.justification)
        LOGGER.debug("  Response:      %s", result.response[:120])

    write_results(output_csv, results)
    LOGGER.info("Wrote %d row(s) to %s", len(results), output_csv)
    LOGGER.info("Chat transcript written to: %s", log_file)
    return output_csv


if __name__ == "__main__":
    final_output_path = run()
    print(f"\nDone. Output written to: {final_output_path}")
