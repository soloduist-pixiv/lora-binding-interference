> **Second revision in progress:** the earlier low-target replay mass does not by itself establish register conditioning. The [second-revision working manuscript](paper/共享LoRA中的条件绑定干扰_第二轮工作稿.md), [new findings](paper/revision2_findings.md) and [registered protocol](paper/revision2_protocol.md) supersede that interpretation. The previous release and its PDF remain historical snapshots. The eight fresh anchors are still being run; partial results are not a completed confirmatory study.

> **Resource interruption and recovery:** job 17 encountered a CUDA allocation failure during validation. Its step-752 checkpoint and all 16 AdamW states were load-verified before same-seed continuation; the failed attempt is retained. The failed process released its resources before any driver reset was performed. See [the incident record](outputs/binding_revision2_20261111/resource_failure_20261111.json) and the recorded queue attempts.

# Factual binding interference during sequential LoRA adaptation

This repository contains the Chinese manuscript, registered review protocol, source code,
canonical answer probability matrices, and CPU analyses for a controlled Cedar–Maple study.
It examines whether old prompts retain key-dependent output structure while losing the old
correct mapping, and tests learning rate, replay, rank, value pairing, and register-label interventions.
The study uses Qwen3-0.6B and SmolLM2-360M-Instruct on a single RTX 4060 Laptop GPU.

## Current second-revision evidence

- Recomputed arithmetic probabilities on the same 16 active keys and test contexts,
  including the blind-mixture bound, signed routing and full probability budgets.
- All 12 matched 640-step continuations and exposure controls are complete.
  On three historical adjacent-pairing anchors, rank-4 replay retains 98.51–99.49%
  correct Cedar mass and 98.92–99.07% correct Maple mass. Its D+M-1 is 0.974–0.986.
  Plain rank 4, plain rank 16 and repeated-Maple controls lose nearly all Cedar mass.
- The eight fresh, support-preserving shuffled anchors have a registered paired
  analysis at high targets. [Every seed and missing pair remains visible](outputs/binding_revision2_20261111/新锚点配对差.md).
- Expanded target modules, a 48-single-token vocabulary, and source-checked temporal
  facts with three sequential registers are registered extensions. Frozen auxiliary
  inputs and probability-analysis CPU checks are complete; expanded-module runners
  have syntax checks only. GPU technical checks and training are pending.

## Historical first-revision observations

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

- [Current working manuscript, second revision](paper/共享LoRA中的条件绑定干扰_第二轮工作稿.md)
- [Current protocol and experiment status](paper/revision2_protocol.md)
- [Historical first-revision manuscript](paper/共享LoRA中的旧事实干扰.md)
- [Historical first-revision typeset PDF](paper/共享LoRA中的旧事实干扰.pdf)
- [Review protocol](paper/review_protocol.md)
- [Historical protocol](paper/validation_protocol.md)
- [Reference audit and correction](paper/references_audit.md)
- [Recorded environment](outputs/binding_review_20261111/environment.json)

## Reproduce figures and statistics on CPU

For the second revision, run `python work/binding_revision2/offline.py`,
`python work/binding_revision2/key_null.py`, `python work/binding_revision2/analyze_training.py`
and `python work/binding_revision2/fixed_report.py` from the repository root.
These use saved matrices and do not import PyTorch. The exact matrix-path lower-bound
audit additionally requires local adapter checkpoints or retraining; those intermediate
checkpoints are not all included in the public archive.

`python work/binding_revision2/aux_data.py` rebuilds the six frozen auxiliary input sets.
Those input files and their source provenance are not experimental results.
Fresh GPU experiments use `serial_jobs.py`; the optional `post_queue.py` waits for the
original 116 jobs before the 32 registered extensions. Do not start a second GPU queue.
`recover_and_continue.py` is a local incident-recovery helper: it requires the saved
optimizer checkpoint, which is not distributed in the completed-only public matrix
package. Publish Git packs after releasing the training model on memory-limited hosts.

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
