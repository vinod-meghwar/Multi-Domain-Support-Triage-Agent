import csv
import logging
import logging.handlers
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


OUTPUT_COLUMNS = [
    "request_type",
    "product_area",
    "decision",
    "justification",
    "response",
]


@dataclass
class TriageResult:
    request_type: str
    product_area: str
    decision: str
    justification: str
    response: str

    def to_dict(self) -> dict:
        return asdict(self)


def setup_logging(log_level: str = "INFO", log_file: Path = None) -> None:
    """Setup logging to both console and log.txt file."""
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
    )


def ensure_project_dirs(project_root: Path) -> None:
    for rel_path in ("docs", "data", "output"):
        (project_root / rel_path).mkdir(parents=True, exist_ok=True)


def load_tickets(csv_path: Path) -> List[str]:
    """Load tickets, combining Issue + Subject columns if both exist."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    cols = [c.strip().lower() for c in df.columns]

    # Combine Issue + Subject for richer context (matches the actual CSV format)
    if "issue" in cols and "subject" in cols:
        issue_col = df.columns[[c.strip().lower() == "issue" for c in df.columns]][0]
        subject_col = df.columns[[c.strip().lower() == "subject" for c in df.columns]][0]
        tickets = [
            f"{str(i).strip()} {str(s).strip()}"
            for i, s in zip(df[issue_col].fillna(""), df[subject_col].fillna(""))
        ]
    elif "query" in cols:
        query_col = df.columns[[c.strip().lower() == "query" for c in df.columns][0]]
        tickets = [str(v).strip() for v in df[query_col].fillna("").tolist()]
    else:
        # Fallback: first column
        tickets = [str(v).strip() for v in df.iloc[:, 0].fillna("").tolist()]

    return [t for t in tickets if t.strip()]


def write_results(csv_path: Path, rows: Iterable[TriageResult]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
