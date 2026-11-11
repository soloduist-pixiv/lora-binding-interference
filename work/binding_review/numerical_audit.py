"""Compare the pilot's preserved cache scoring with re-evaluated tree scoring."""
import json

import numpy as np

from offline import OUT, entropy


def measurements(old, maple, indices):
    selected = indices % 6 < 2
    old = old[selected].reshape(24,2,96).mean(1)
    maple = maple[selected].reshape(24,2,96).mean(1)
    p = np.c_[old, np.maximum(0,1-old.sum(1))]
    q = np.c_[maple, np.maximum(0,1-maple.sum(1))]
    return np.array([
        entropy(p.mean(0))-entropy(p).mean(),
        old[np.arange(24), np.arange(0,48,2)].mean(),
        (entropy((p+q)/2)-(entropy(p)+entropy(q))/2).mean(),
    ])


def main():
    folder = OUT / "runs/Qwen3-0.6B_s107_adjacent_lr0.0003_r4"
    comparison = []
    for path in sorted((folder / "cache_reference").glob("s*.npz")):
        with np.load(path) as old, np.load(folder/path.name) as new:
            a = measurements(old["old"], old["maple"], old["indices"])
            b = measurements(new["old"], new["maple"], new["indices"])
        comparison.append(dict(step=int(path.stem[1:]), information_difference=float(b[0]-a[0]),
                               correct_mass_difference=float(b[1]-a[1]), js_difference=float(b[2]-a[2])))
    target = OUT / "numerical_audit.json"
    target.write_text(json.dumps(comparison, indent=2), encoding="utf8")
    for field in ("information_difference", "correct_mass_difference", "js_difference"):
        print(field, max(abs(row[field]) for row in comparison))


if __name__ == "__main__":
    main()
