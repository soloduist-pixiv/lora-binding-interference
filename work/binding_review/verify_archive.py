"""Verify the published snapshot before regenerating analyses or figures."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    records = json.loads((ROOT/"manifest.json").read_text(encoding="utf8"))
    failed = []
    for row in records:
        path = ROOT/row["path"]
        if (not path.is_file() or path.stat().st_size != row["size"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            failed.append(row["path"])
    if failed:
        raise SystemExit("Missing or modified snapshot files:\n"+"\n".join(failed))
    print(f"Verified {len(records)} files and {sum(r['size'] for r in records):,} bytes.")


if __name__ == "__main__":
    main()
