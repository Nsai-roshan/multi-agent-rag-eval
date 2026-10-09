import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Code" / "v2" / "eval" / "show_item.py"


def test_show_item_list_prints_all_golden_ids():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line for line in result.stdout.splitlines() if line.startswith("v")]
    assert len(rows) == 80
    assert "v1-01" in result.stdout
    assert "v2-30" in result.stdout


def test_show_item_prints_full_resolved_chunk():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "v1-02"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "What are some possible negatove side effects" in result.stdout
    assert "FULL CHUNK TEXT" in result.stdout
    assert "Creatine" in result.stdout
