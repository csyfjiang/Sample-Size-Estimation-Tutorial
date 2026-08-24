# Sample-Size-Estimation-Tutorial

Sample-size and precision calculators for **clinical prediction models with a binary
outcome** (e.g. pCR), covering the whole study lifecycle:

> **Development** → **External validation** → **Head-to-head model comparison**

It ships as a small Python package (`sampsizeval`), a command-line tool, a **zero-install
web demo** that runs entirely in your browser (Pyodide — *your data never leaves your
machine*), and a 20-minute Beamer talk.

**🔗 Live demo:** https://csyfjiang.github.io/Sample-Size-Estimation-Tutorial/

---

## Why not just use "10 events per variable"?

Rules of thumb (10 EPV; "100 events + 100 non-events" for validation) ignore that the
*required* sample size depends jointly on the **event fraction**, the **number of
predictors**, the **expected performance** (R², c-statistic) and the **target
precision**. Recent methodology (Riley, Snell, van Smeden, Collins, Jung) replaces the
rules with criterion-based calculations. This repo implements four of them and, crucially,
runs each in **reverse-lookup** mode: *plug in your actual cohort size → read off the
achieved precision / power.*

| Stage | Question | Method | Module |
|---|---|---|---|
| Development | Is the training set big enough to fit a *stable* model? | Riley et al., *BMJ* 2020;368:m441 | `development` |
| Validation (simulation) | Is validation precise enough? (allows miscalibration) | Snell et al., *J Clin Epidemiol* 2021;135:79–89 | `validation_sim` |
| Validation (closed form) | Is validation precise enough? (fast, `pmvalsampsize`-style) | Riley et al. (part 3), *BMJ* 2023;383:e074821 | `validation_closed` |
| Comparison | Is B *significantly* better than A? (DeLong-equivalent) | Jung, *Pharm Stat* 2024;23(4):557–569 | `compare_auc` |

---

## Install

```bash
git clone https://github.com/csyfjiang/Sample-Size-Estimation-Tutorial.git
cd Sample-Size-Estimation-Tutorial
pip install -e .          # installs numpy + scipy
pip install -e ".[csv,test]"   # optional: pandas (CSV) + pytest
```

## Quickstart (Python)

```python
import sampsizeval as ssv

# 1. Development: is N=717 enough? (phi=event fraction, P=candidate parameters)
r2 = 0.15 * ssv.development.max_r2cs(0.30)
d = ssv.development.reverse_lookup_development(n=717, phi=0.30, P=20, r2cs=r2)
print(d["overall_required_n"], d["binding_criterion"], d["sufficient"])
# -> ~1599  B3  False   (shrinkage is the binding constraint)

# 2. Validation precision at N=334
v = ssv.validation_closed.reverse_lookup_binary(n=334, phi=0.30, c_stat=0.75)
print(v["c_stat_ci_width"], v["calibration_slope_ci_width"])
# -> ~0.11 (c, OK)   ~0.5 (slope, wide -> the binding one)

# 3. Compare two AUCs / power at your N
c = ssv.compare_auc.required_sample_size(theta1=0.75, theta2=0.82, rho=0.5, gamma=0.30)
p = ssv.compare_auc.achieved_power(N=334, theta1=0.75, theta2=0.82, rho=0.5, gamma=0.30)
print(c["N"], p["power"])   # -> 422   0.70
```

## Quickstart (CLI)

```bash
sampsizeval dev        --n 717 --phi 0.30 --P 20
sampsizeval val-closed --n 334 --phi 0.30 --c 0.75
sampsizeval val-sim    --n 334 --mu -0.85 --sigma 1.0
sampsizeval compare    --n 334 --theta1 0.75 --theta2 0.82 --rho 0.5 --gamma 0.30

# Estimate the inputs straight from patient-level data:
sampsizeval from-csv examples/predictions.csv           # -> phi, c-stat, mu, sigma
sampsizeval from-csv examples/two_models.csv --two      # -> theta1, theta2, rho, gamma
```

---

## Drive in a CSV instead of guessing parameters

Both the CLI (`from-csv`) and the web demo can estimate the calculator inputs from a real
dataset. Two supported layouts (see [`examples/`](examples/)):

**`predictions.csv`** — one model's predictions (feeds validation calculators):

| column | meaning |
|---|---|
| `predicted_prob` | model's predicted probability p̂ᵢ |
| `true_outcome` | observed outcome (1/0) |

→ estimates `phi` (event fraction), `c_stat` (AUC) and `mu`/`sigma` of the linear predictor.

**`two_models.csv`** — two models on the *same* patients (feeds the comparison calculator):

| column | meaning |
|---|---|
| `model_A_score` | model A predicted score/probability |
| `model_B_score` | model B predicted score/probability |
| `true_outcome` | observed outcome (1/0) |

→ estimates `theta1`, `theta2` (each AUC), `rho` (score correlation) and `gamma`.

---

## Web demo

The [`docs/`](docs/) folder is a static site that loads
[Pyodide](https://pyodide.org) and runs the exact same `sampsizeval` code in the browser
— four tabs, manual inputs, and drag-and-drop CSV. **Nothing is uploaded**; all
computation is client-side.

Enable it on GitHub: **Settings → Pages → Source: `Deploy from a branch` → `main` /
`docs`**. The demo then lives at
`https://<user>.github.io/Sample-Size-Estimation-Tutorial/`.

> The browser copy of the package lives in `docs/py/sampsizeval/`. If you edit the
> top-level `sampsizeval/`, re-sync with `python scripts/sync_docs.py`.

---

## Slides

A ready-to-present 20-minute Beamer deck is in [`slides/main.tex`](slides/main.tex)
(metropolis theme; compiles on Overleaf with pdfLaTeX, or XeLaTeX for Fira fonts).
It walks through the lifecycle logic, each method's key formulas, worked numbers, and
the TRIPOD+AI item-10 manuscript templates.

---

## Manuscript rationale (TRIPOD+AI item 10)

The repo bundles drop-in sample-size-rationale paragraphs for development, validation and
comparison. See the tutorial ([`sample_size_tutorial.md`](sample_size_tutorial.md)) and
the slides.

## Tests

```bash
pytest -q
```

## Citing

This is an implementation of published methods — please cite the original papers (above)
and, if useful, this repository. A [`CITATION.cff`](CITATION.cff) is provided.

## Disclaimer

Educational / planning tool. For numbers going into a manuscript, confirm against the
official `pmsampsize` / `pmvalsampsize` R packages. Some calculations (calibration slope
in the closed-form module; the c→R²cs bridge) are approximations, flagged in the code.

## License

MIT — see [LICENSE](LICENSE).
