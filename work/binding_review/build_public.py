"""Curate a standalone publication repository from explicit research files."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "publish/lora-binding-interference"
REPO = "https://github.com/soloduist-pixiv/lora-binding-interference"
COPIED = set()
SOURCE_FILES = {
    "round4": ("core.py", "compare_models.py", "train.py", "checks.py"),
    "round5": ("pilot.py", "codebook.py"),
    "binding_constraints": ("run.py",),
    "binding_followup": ("run.py",),
    "update_diagnostics": ("diagnostics.py", "extra.py"),
    "prospective_updates": ("experiment.py", "stream.py", "disjoint_mass.py", "paper_reanalysis.py",
                            "binding_plane.py", "stage0_power.py", "stage1_report.py"),
    "binding_review": ("cached_score.py", "verify_scorer.py", "run.py", "pipeline.py",
                       "offline.py", "analyze.py", "figures.py", "tables.py", "numerical_audit.py",
                       "capacity_reference.py","support_reference.py","routes.py","report.py",
                       "typeset.py","build_public.py","verify_archive.py","verify_extended.py","rescore.py"),
}


def copy(path):
    target = DEST / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    COPIED.add(target)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for directory, names in SOURCE_FILES.items():
        for name in names:
            path = ROOT / "work" / directory / name
            if path.is_file():
                copy(path)
    for pattern in ("*.md", "*.json", "*.png", "*.pdf", "*.tex"):
        for path in (ROOT / "paper").glob(pattern):
            copy(path)
    source = ROOT / "outputs/model_compare_open_20261108/prospective_updates"
    for seed in (29, 43, 107, 108, 109):
        for name in ("mass.json", "mass_prefix_control.json", "mass_cpu_reference.json",
                     "old.pt", "old_gate.json"):
            path = source / f"disjoint_mass/s{seed}" / name
            if path.is_file():
                copy(path)
    for seed in range(101, 107):
        copy(source / f"s{seed}/stage1_trajectories.json")
    copy(source / "stage1_gate.json")
    copy(ROOT / "outputs/model_compare_open_20261108/binding_followup/audit.json")
    review = ROOT / "outputs/binding_review_20261111"
    for path in review.rglob("*"):
        if path.is_file() and (path.suffix in (".json", ".npz", ".png", ".pdf")
                               or path.name == "old.pt"):
            copy(path)
    pilot = review / "runs/Qwen3-0.6B_s107_adjacent_lr0.0003_r4"
    for step in (16,64):
        copy(pilot / f"adapter_{step:04d}.pt")
    representative_adapters = []
    for name in ("Qwen3-0.6B_s211_alphabetic_lr0.0003_r4",
                 "SmolLM2-360M-Instruct_s211_adjacent_lr0.0003_r4"):
        folder = review/"runs"/name
        complete = folder/"complete.json"
        if complete.exists():
            step = json.loads(complete.read_text(encoding="utf8"))["last_step"]
            checkpoint = folder/f"adapter_{step:04d}.pt"
            copy(checkpoint)
            representative_adapters.append(checkpoint.relative_to(ROOT).as_posix())
    environment = json.loads((review / "environment.json").read_text(encoding="utf8"))
    packages = environment["packages"]
    requirements = "\n".join(f"{name}=={packages[name].split('+')[0]}" for name in
                             ("torch", "transformers", "accelerate", "safetensors",
                              "numpy", "scipy", "matplotlib", "tokenizers", "huggingface-hub"))
    (DEST / "requirements.txt").write_text(requirements + "\nmarkdown-it-py==4.0.0\n", encoding="utf8")
    (DEST / "requirements-analysis.txt").write_text(
        "\n".join(f"{name}=={packages[name]}" for name in ("numpy","matplotlib")) + "\n",
        encoding="utf8")
    (DEST / ".gitignore").write_text(
        "Models/\n__pycache__/\n*.pyc\n**/state.pt\n**/adapter_*.pt\n"
        "!outputs/binding_review_20261111/runs/Qwen3-0.6B_s107_adjacent_lr0.0003_r4/adapter_0016.pt\n"
        "!outputs/binding_review_20261111/runs/Qwen3-0.6B_s107_adjacent_lr0.0003_r4/adapter_0064.pt\n"
        "publish/\n*.tmp\n*.log\n" +
        "".join("!"+path+"\n" for path in representative_adapters), encoding="utf8")
    readme = """# Factual binding interference during sequential LoRA adaptation

This repository contains the Chinese manuscript, registered review protocol, source code,
canonical answer probability matrices, and CPU analyses for a controlled Cedar–Maple study.
It examines whether old prompts retain key-dependent output structure while losing the old
correct mapping, and tests learning rate, replay, rank, value pairing, and register-label interventions.
The study uses Qwen3-0.6B and SmolLM2-360M-Instruct on a single RTX 4060 Laptop GPU.

## Main observations

- A fixed early-step information rebound depends on learning rate and learning stage.
  All three historical Qwen anchors rebound from step 16 to 64 at lr=1e-3; none
  does so at lr=1e-4.
- At the same Maple validation mean log probability of -0.5, correct Cedar mass
  averages 25.12% at lr=1e-4 and 12.67% at lr=1e-3. Rank-4 replay at lr=3e-4
  retains 66.07%. These are three-seed descriptive means.
- SmolLM2 passes the old-task learning gate for all three seeds. Its fixed early
  window has no rebound, while an auxiliary later-progress comparison shows
  increasing key information alongside decreasing correct Cedar mass.
- Alphabetic values have substantial probability outside the canonical candidate
  set. Coarsened information and label JS have limited interpretation there.

## Read the study

- [Manuscript](paper/共享LoRA中的旧事实干扰.md)
- [Typeset PDF](paper/共享LoRA中的旧事实干扰.pdf)
- [Review protocol](paper/review_protocol.md)
- [Historical protocol](paper/validation_protocol.md)
- [Reference audit and correction](paper/references_audit.md)
- [Recorded environment](outputs/binding_review_20261111/environment.json)

## Reproduce figures and statistics on CPU

Install requirements-analysis.txt. PyTorch and a GPU are not needed for
the following analysis entry points.

~~~sh
python work/binding_review/offline.py
python work/binding_review/analyze.py
python work/binding_review/figures.py
python work/binding_review/routes.py
python work/binding_review/tables.py
~~~

The offline entry point corrects historical permutation references, candidate coverage,
routing profiles, and gap-closure comparisons. The analyzer rebuilds fixed-step and
validation-selected progress measurements from saved NPZ matrices. Every NPZ contains
absolute canonical event masses for Cedar and Maple labels, validation log probabilities,
and evaluated row indices. The remaining probability is preserved as an other category.

## Repeat model experiments

Use a compatible CUDA build of PyTorch. Place the model snapshots in the following
directories before running the fully offline training scripts.

- Models/Qwen3-0.6B from https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca
- Models/SmolLM2-360M-Instruct from https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/tree/a10cc1512eabd3dde888204e902eca88bddb4951

The recorded environment used Python 3.14.3, PyTorch 2.10.0+cu130, Transformers 5.10.2,
Windows 11, approximately 16 GiB RAM, and an 8188 MiB RTX 4060 Laptop GPU.
Jobs require 6 GiB initially free GPU memory and enforce a 6 GiB reserved-memory cap.

~~~sh
python work/binding_review/verify_scorer.py
python work/binding_review/verify_extended.py
python -u work/binding_review/pipeline.py --output-root outputs/reproduction
~~~

The six later exploratory runs can be repeated after the main queue, one at a time:

~~~sh
python -u work/binding_review/run.py --seed 107 --historical --rank 16 --replay --minimum-steps 640 --output-root outputs/reproduction
python -u work/binding_review/run.py --seed 108 --historical --rank 16 --replay --minimum-steps 640 --output-root outputs/reproduction
python -u work/binding_review/run.py --seed 109 --historical --rank 16 --replay --minimum-steps 640 --output-root outputs/reproduction
python -u work/binding_review/run.py --seed 211 --pairing support_permuted --output-root outputs/reproduction
python -u work/binding_review/run.py --seed 223 --pairing support_permuted --output-root outputs/reproduction
python -u work/binding_review/run.py --seed 227 --pairing support_permuted --output-root outputs/reproduction
~~~

The supervisor runs jobs serially and records failed jobs. The output-root option keeps
new training results separate from the published measurements. A complete job is skipped.
Incomplete new-task runs resume from local state checkpoints if available. All published
matrices can be reanalyzed without training. The publication snapshot includes synthetic
Cedar anchors, and excludes base-model weights, transient optimizer state, and dense
intermediate adapter files. Those files are generated during training.
The historical rescore.py audit needs those locally generated dense checkpoints;
the published default pipeline already uses FP32 evaluation.

The registered factorial seeds are 211, 223, and 227. Learning-rate, replay, and rank controls
use historical Cedar anchors 107, 108, and 109. Numeric and alphabetic targets are
generated by code and saved in each run's protocol.json.

## Scope

The benchmark measures controlled code recall. Register JS measures behavior under a
string-label intervention. See the manuscript for coverage, incomplete comparisons,
and the distinction between historical, registered, and exploratory evidence.

## Files and scoring

The file-level SHA-256 manifest excludes the manifest itself and Git internals.
Verify the pristine snapshot with python work/binding_review/verify_archive.py
before regenerating outputs (figure PDFs contain new creation timestamps).
Model fingerprints identify the exact local snapshots used. Base weights must be
downloaded from the official model repositories, subject to their own terms.
The source.json fingerprint refers to local acquisition provenance, which is not
required to load the model; the weight, config, and tokenizer fingerprints identify
the load-bearing files.

Each run has protocol.json (seed, values, split IDs, tokenization, hyperparameters),
trace.json (validation scores, sample counts and memory), and sXXXX.npz matrices.
The candidate columns are 48 Cedar values followed by 48 Maple values, in protocol
order. NPZ indices address old_eval_ids, ordered by key and then six test contexts.
Dense progress analysis uses the first two contexts of each key. There is no
renormalization over candidate columns: one minus their sum is the other bucket.
Saved old.pt files contain only synthetic-task adapter parameters.

Figure/statistic scripts read the published default output directory. To analyze
a newly trained output-root, use a separate checkout and replace that checkout's
default results directory with your reproduction directory. Keep the published
archive intact for comparison.

The PDF can be rebuilt from paper/manuscript.tex with XeLaTeX. Regenerating the TeX
from Markdown additionally requires markdown-it-py; run work/binding_review/typeset.py.
"""
    (DEST / "README.md").write_text(readme, encoding="utf8")
    (DEST / ".gitattributes").write_text("* -text\n",encoding="utf8")
    included = COPIED | {DEST/name for name in
                         ("README.md","requirements.txt","requirements-analysis.txt",
                          ".gitignore",".gitattributes")}
    for path in included:
        if path.suffix in (".md",".py",".json",".txt",".tex") or path.name.startswith(".git"):
            path.write_bytes(path.read_bytes().replace(b"\r\n",b"\n"))
    files = []
    for path in sorted(included):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(dict(path=path.relative_to(DEST).as_posix(), size=path.stat().st_size, sha256=digest))
    (DEST / "manifest.json").write_bytes(json.dumps(files, indent=2).encode("utf8"))
    print(DEST, len(files), sum(row["size"] for row in files), "bytes", REPO)


if __name__ == "__main__":
    main()
