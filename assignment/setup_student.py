"""Run once:  python setup_student.py 123456M

Stores your student number in ./STUDENT_ID and prints your personal CartPole variant.
Every script reads it from there. Do NOT use someone else's number: all runs carry a
fingerprint of the ID they were generated with, and submissions are re-run and checked.
"""
import sys

from rl_assignment.student import ID_FILE
from rl_assignment.variant import make_variant

if len(sys.argv) != 2:
    sys.exit(__doc__)
v = make_variant(sys.argv[1])
ID_FILE.write_text(v.student_id + "\n")
print(v.card())
print(f"\nSaved to {ID_FILE}. Copy the card above into A1 of your report.")
