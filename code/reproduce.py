"""Re-run of the re-implementation: Petri nets, GOAP and a genetic-algorithm policy in the corridor model (no returns).
Uses the calibrated parameters stored in _v10.json and 100 fresh seeds (100-199).
Usage: python reproduce.py <output_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_v10 as F  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
OUT.mkdir(parents=True, exist_ok=True)
e, prm, rows = json.loads(Path(__file__).with_name("_v10.json").read_text(encoding="utf-8"))
res = F.run(prm, range(100, 200))
LABEL = {"Петри": "Petri nets", "GOAP": "GOAP", "Генетика": "Genetic algorithm"}
TARGET = F.T  # (mean, SD) read from the paper figures: (critical, non-critical)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
rs = np.random.default_rng(1)
lines = ["| Method | Critical: model mean ± SD | Critical: target | Non-critical: model mean ± SD | Non-critical: target | N |", "|---|---|---|---|---|---|"]
for j, n in enumerate(F.T):
    cr, og = res[n]
    lines.append(f"| {LABEL[n]} | {cr.mean():.1f} ± {cr.std():.1f} | {TARGET[n][0][0]:.0f} ± {TARGET[n][0][1]:.1f} | {og.mean():.1f} ± {og.std():.1f} | {TARGET[n][1][0]:.0f} ± {TARGET[n][1][1]:.1f} | {len(cr)} / {len(og)} |")
    c2, o2 = rs.choice(cr, 200, replace=True), rs.choice(og, 200, replace=True)
    bins = np.linspace(0, max(np.percentile(c2, 99.5), np.percentile(o2, 99.5)) + 2, 28)
    ax[j].hist(o2, bins=bins, color="#3b5bdb", alpha=0.65, label="Non-critical")
    ax[j].hist(c2, bins=bins, color="#e03131", alpha=0.65, label="Critical")
    ax[j].set_title(f"{LABEL[n]}: crit. {cr.mean():.1f}±{cr.std():.1f}, non-crit. {og.mean():.1f}±{og.std():.1f}")
    ax[j].set_xlabel("Time, model units (minutes)")
ax[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "forms_model.png", dpi=110, bbox_inches="tight")
(OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
