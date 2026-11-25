"""Where the student ID comes from: the STUDENT_ID file in the repository root
(created once with `python setup_student.py <your id>`) or the CCE5502_STUDENT_ID environment variable."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from .variant import Variant, make_variant

ROOT = Path(__file__).resolve().parent.parent
ID_FILE = ROOT / "STUDENT_ID"


def load_variant() -> Variant:
    sid = os.environ.get("CCE5502_STUDENT_ID") or (ID_FILE.read_text().strip() if ID_FILE.exists() else "")
    if not sid:
        sys.exit("No student ID set. Run once:  python setup_student.py <your UM student number>")
    return make_variant(sid)
