"""Publish an explicit completed-only supplement; never copy optimizer/base weights."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/"publish/lora-binding-interference"
OUT=ROOT/"outputs/binding_revision2_20261111"

def main():
    files=list((ROOT/"work/binding_revision2").glob("*.py"))
    files += [ROOT/"paper"/n for n in ("revision2_protocol.md","revision2_reference_audit.md",
              "revision2_findings.md","reference_identifiers.md","references_audit.md",
              "共享LoRA中的条件绑定干扰_第二轮工作稿.md")]
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in (".md",".json",".png",".pdf")]
    files += list((OUT/"auxiliary_data").glob("*.json"))
    for complete in (OUT/"auxiliary").glob("*/*/stage*/complete.json"):
        files += [p for p in complete.parent.iterdir() if p.suffix in (".json",".npz")]
        files += [complete.parent.parent/"protocol.json"]
    for anchor in (OUT/"auxiliary").glob("*/anchor/old.pt"):
        files += [anchor]+list(anchor.parent.glob("*.json"))
    for folder in (OUT/"runs").iterdir():
        if (folder/"complete.json").exists():
            files += [p for p in folder.iterdir() if p.suffix in (".json",".npz")]
    for folder in (OUT/"anchors").iterdir() if (OUT/"anchors").exists() else []:
        if (folder/"old.pt").exists():
            files += [p for p in folder.iterdir() if p.suffix==".json" or p.name=="old.pt"]
    previous=json.loads((REPO/"manifest.json").read_text())
    paths={r["path"] for r in previous}
    for source in files:
        rel=source.relative_to(ROOT); target=REPO/rel
        target.parent.mkdir(exist_ok=True,parents=True)
        data=source.read_bytes()
        if source.suffix in (".json",".md",".py"):data=data.replace(b"\r\n",b"\n")
        if source.suffix==".json":json.loads(data)
        target.write_bytes(data);paths.add(rel.as_posix())
    # This wrapper was renamed to avoid shadowing Python's standard queue module.
    old=REPO/"work/binding_revision2/queue.py"
    if old.exists():
        old.unlink(); paths.discard("work/binding_revision2/queue.py")
    rows=[dict(path=s,size=(REPO/s).stat().st_size,
               sha256=hashlib.sha256((REPO/s).read_bytes()).hexdigest()) for s in sorted(paths)]
    (REPO/"manifest.json").write_bytes(json.dumps(rows,indent=2).encode())
    readme=REPO/"README.md"
    readme.write_bytes(readme.read_bytes().replace(b"\r\n",b"\n"))
    # Recalculate if a previously tracked README was normalized above.
    for row in rows:
        if row["path"]=="README.md":
            row.update(size=readme.stat().st_size,sha256=hashlib.sha256(readme.read_bytes()).hexdigest())
    (REPO/"manifest.json").write_bytes(json.dumps(rows,indent=2).encode())
    subprocess.run(["git","add","README.md","manifest.json","paper","work/binding_revision2",
                    "outputs/binding_revision2_20261111"],cwd=REPO,check=True)
    result=subprocess.run(["git","diff","--cached","--check"],cwd=REPO,capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stdout[:2000])
    result=subprocess.run(["git","commit","-m","research: publish completed matched-budget controls"],cwd=REPO,capture_output=True,text=True)
    if result.returncode:raise RuntimeError((result.stdout+result.stderr)[-2000:])
    result=subprocess.run(["git","-c","pack.threads=1","-c","pack.windowMemory=32m",
                           "-c","pack.deltaCacheSize=16m","-c","pack.window=1",
                           "push","origin","main"],cwd=REPO,capture_output=True,text=True)
    if result.returncode:raise RuntimeError((result.stdout+result.stderr)[-2000:])
    print("Published",len(rows),"manifest files; completed branches only.")

if __name__=="__main__":main()
