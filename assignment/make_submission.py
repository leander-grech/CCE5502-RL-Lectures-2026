"""Build your submission ZIP:   python make_submission.py Firstname_Lastname

Put your report as report.pdf in this folder first. The ZIP (Firstname_Lastname_CCE5502.zip) contains:
    report.pdf                     your answers, labelled A1, A2, ... E2, with your plots
    runs/                          every run: config.json, monitor.csv, eval*.json, models, tb/ (TensorBoard logs)
    videos/  traces/  figures/     the files you refer to in the report
    STUDENT_ID  git_log.txt        (git_log.txt is written for you)
    rl_assignment/                 including your rewards.py
Then upload the ZIP to the VLE. Your .venv/ is never included.
"""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INCLUDE = ["report.pdf", "STUDENT_ID", "git_log.txt", "rl_assignment", "runs", "videos", "traces", "figures"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    name = re.sub(r"[^0-9A-Za-z_-]", "_", sys.argv[1].strip())
    out = ROOT / f"{name}_CCE5502.zip"
    problems = []
    if not (ROOT / "report.pdf").exists():
        sys.exit("report.pdf not found: save your report as report.pdf in this folder, then run this again.")
    if not (ROOT / "STUDENT_ID").exists():
        problems.append("no STUDENT_ID file (run python setup_student.py <your id>)")
    try:
        log = subprocess.check_output(["git", "log", "--pretty=format:%h %ad %s", "--date=iso"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        (ROOT / "git_log.txt").write_text(log + "\n")
    except Exception:
        problems.append("could not write git_log.txt (is git installed, and is this the cloned repository?)")

    runs = sorted(p.parent for p in (ROOT / "runs").glob("*/seed*/config.json")) if (ROOT / "runs").exists() else []
    if not runs:
        problems.append("no runs found in runs/")
    no_tb = [str(r.relative_to(ROOT)) for r in runs if not list(r.glob("tb/**/events.out.tfevents*"))]
    if no_tb:
        problems.append(f"runs without TensorBoard logs: {', '.join(no_tb)}")
    for d in ["videos", "figures"]:
        if not (ROOT / d).exists() or not any((ROOT / d).iterdir()):
            problems.append(f"{d}/ is empty or missing")

    files = []
    for item in INCLUDE:
        p = ROOT / item
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += [f for f in p.rglob("*") if f.is_file() and not SKIP_PARTS & set(f.parts) and f.suffix != ".pyc"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path(f"{name}_CCE5502") / f.relative_to(ROOT))

    print(f"wrote {out.name}: {len(files)} files, {len(runs)} runs, {out.stat().st_size / 1e6:.1f} MB")
    for p in problems:
        print(f"  check: {p}")
    print("Open the ZIP to check that report.pdf and runs/ are inside, then upload it to the VLE.")


if __name__ == "__main__":
    main()
