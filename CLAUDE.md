# CLAUDE.md — Project Guide

This repo contains the analysis code for a discrete-choice modelling paper currently in **Round 2 revision** for the *Transportation* journal: perceived traffic safety's effect on cycling route choice, built from street-view imagery + a stated-preference panel. When you pick this up, read `PLAN.md` for the immediate work-in-progress handoff, then come back here for the durable context.

## Dataset invariants

After `apply_data_cleaning(drop_problematic_rid=True)`:

| | Count |
|---|---|
| Total rows | **11,190** |
| Individuals | **746** |
| Train individuals (from labelled `train` column) | **606** |
| Test individuals (from labelled `test` column) | **140** |
| Observations per individual | 15 |

Raw CSV (`data/raw/cv_dcm.csv`) has 11,289 rows / 752 individuals. Cleaning drops the last 15 rows of RID 63 (flagged by the co-author as duplicates + stray SEQ row) and then drops any RID with fewer than 15 observations. The function in [cycling_safety_svi/modeling/mxl_functions.py](cycling_safety_svi/modeling/mxl_functions.py) asserts these counts — if you refactor it, keep the assertion.

The `train` and `test` columns are **pre-labelled** in `cv_dcm.csv` and that labelling is canonical. Do not re-split randomly. (The old `stepwise_train_test.py` did a random 80/20 and produced 597/150; it has been changed to use the labelled columns.)

## Data is not in git

`data/` is in `.gitignore`. Total ~92 MB across:

```
data/raw/
  cv_dcm.csv                          596 KB   # panel choice data (canonical)
  database_2024_10_07_135133.db       24 MB    # demographics SQLite
  df_choice_with_Vimg.csv             1.3 MB   # original utility computations (Terra 2024)
  main_design.csv                     42 MB    # experiment design matrix
  segmented_images.csv                880 KB
  ...
data/processed/
  predicted_danish/cycling_safety_scores.csv    # safety scores per image
  segmentation_results/pixel_ratios.csv         # Mask2Former segmentation ratios
  model_results/image_utilities_model4.csv      # written by compute_model4_utilities.py
```

Transfer `data/` out-of-band (scp/rsync) whenever you move the repo to a new machine.

## How to run the modeling scripts

The modeling scripts use **bare imports** (`from mxl_functions import ...`), not package-style imports. Two things must be true at runtime:

1. `cycling_safety_svi/modeling/` is on `PYTHONPATH`
2. CWD is the project root (so relative paths like `data/raw/cv_dcm.csv` resolve)

Canonical invocation:
```bash
PYTHONPATH=$PWD/cycling_safety_svi/modeling \
  .venv/bin/python -u -c "from <script_basename> import main; main()"
```

Use `.venv/bin/python` directly — it's uv-managed Python 3.10.16. Don't `source .venv/bin/activate` unless you're working interactively; the direct path is more reliable in background jobs.

Use `-u` (unbuffered stdout) for any background run so the log file grows in real time. Biogeme prints important progress to stdout.

Default `__main__` blocks in these scripts are **not** safe — they point at old checkpoint directories:
- `choice_model_benchmark.py`'s `__main__` calls `main(checkpoint_dir='.../mxl_choice_20250725_122947')`, which is the pre-revision run. Always invoke `main()` with no kwarg (fresh run) or pass the current dated directory.
- `safety_demographics_interaction_model.py` CLI defaults `--checkpoint` to `.../safety_demographics_20251028_174315`, also pre-revision. Always pass `--checkpoint ""` for a fresh timestamped dir, or pass the current one to resume.

## The model scripts and their inputs/outputs

All outputs land under `reports/models/` (inside the Overleaf-synced submodule — see *reports/ is an Overleaf submodule* below).

| Script | What it does | Reads | Writes |
|---|---|---|---|
| `choice_model_benchmark.py` | Backward elim + 4 main MXL (Base / +Safety / +Seg / Full = Model 4) + 2 WTP MXL (`safety_vs_tt`, `safety_vs_tl`) + MNL WTP comparisons | `data/raw/cv_dcm.csv`, `data/processed/predicted_danish/cycling_safety_scores.csv`, `data/processed/segmentation_results/pixel_ratios.csv` | `reports/models/mxl_choice_<timestamp>/` |
| `stepwise_train_test.py` | Train/test generalisation check using the labelled split columns | same | `reports/models/stepwise_train_test_<timestamp>/` |
| `compute_model4_utilities.py` | Per-image utility scores from Model 4 | `final_full_model.pickle`, safety scores, segmentation | `data/processed/model_results/image_utilities_model4.csv` |
| `safety_demographics_interaction_model.py` | 17 safety × demographic interaction models (MXL Method 1: shared sigma) | `final_full_model.pickle`, `database_2024_10_07_135133.db` | `reports/models/interaction/safety_demographics_<timestamp>/<group>/` |
| `compute_lrt_all_models.py` | LRT each interaction model vs Model 4 | **hardcoded** LL/params values in the source — refresh from new .tex files | stdout table |
| `extract_all_interaction_params.py` | Parse interaction .tex files → JSON of safety-parameter z-scores | interaction `.tex` outputs | `all_interaction_params.json` (path hardcoded — patch before use) |
| `verify_interaction_zscores.py` | Sanity-check the z-score arithmetic with hardcoded old betas | nothing | stdout; needs manual update after reruns |

The last three are lightweight and run in seconds. The first three are heavy (~1-6 hours each).

## Biogeme quirks

### Silent exit after the "low draws" warning

MXL estimations in this project occasionally terminate the Python process with **exit code 0 and no traceback, no stderr**, immediately after biogeme prints:

```
The number of draws (1000) is low. The results may not be meaningful.
```

The warning itself is *not* the cause — most runs print it and continue fine. But some fraction of the time, the process just dies there. It is non-deterministic and retry-safe:

- ChoiceModelBenchmark has a checkpoint mechanism: `main(checkpoint_dir='reports/models/mxl_choice_<ts>')` skips any already-pickled model.
- SafetyDemographicsInteractionModel has a checkpoint mechanism: `--checkpoint <dir>` skips any group that already has a `demographics_interaction_model_<group>.tex` summary file.
- For the demographics run specifically, [scripts/run_demographics_with_retry.sh](scripts/run_demographics_with_retry.sh) wraps the script in a bash loop that re-runs with the same checkpoint until all 17 summaries exist (or 20 attempts fail).

**Verify after every MXL-heavy background run** that the expected pickle / tex / html files actually landed. Exit code 0 alone does not mean success.

### Real Biogeme C++ bug at WTP `safety_vs_tl`

Different failure mode, genuinely buggy biogeme. Seen in `choice_model_benchmark.py`'s WTP phase:

```
RuntimeError: src/cythonbiogeme/cpp/biogeme.cc:449: Biogeme exception:
  Error for data entry 635 : Truck_1_14: Value 2199023255979 out of range [0,434]
```

Decoding: at row 635 of the wide panel (746 rows = individuals), the C++ layer interpreted the *value* of the `Truck_1_14` variable as an *index* into the 435-column wide dataset. `2,199,023,255,979 ≈ 2^41` is classic uninitialised-memory signature (reading stack garbage as int64). This is deterministic on this data + draw seed. Workarounds, in order:

1. Retry with `main(checkpoint_dir=<new dir>)` — sometimes biogeme's memory layout differs across runs and the bug doesn't trigger.
2. Bump `self.num_draws = 1000` to `2000` in `ChoiceModelBenchmark.__init__` — changes memory layout, often makes the bug go away.
3. Compute WTP for `safety_vs_tl` as the ratio `β_safety / β_TL` from Model 4 directly, and document the workaround in the response letter.

## `reports/` is an Overleaf-linked submodule

`reports/` is a git submodule backed by its own repo (`cycling_safety_perception_report`). The submodule is synced bidirectionally with Overleaf — Overleaf pushes commits to GitHub, we pull them locally.

**Hard rule:** `cd reports && git pull` before editing any `.tex` file in that directory. If you edit without pulling, you can cause merge conflicts that corrupt the Overleaf document. The user has stated this multiple times; respect it.

Other rules (full detail in [reports/CLAUDE.md](reports/CLAUDE.md)):
- `.aux`, `.bbl`, `.log`, `.fdb_latexmk`, `.fls`, `.out`, `.spl` files are LaTeX build artifacts — leave them untracked.
- Model output directories under `reports/models/` are written by this project's modeling pipeline. Leave old timestamped dirs alone (they are "before revision" snapshots useful for the R&R response letter).
- **30 MB file size cap** — Overleaf breaks at a 30 MB project size. A pre-commit hook at `.git/modules/reports/hooks/pre-commit` blocks commits with oversized files. If a figure is large, compress it first.
- There's a detailed **response letter protocol** in `reports/CLAUDE.md` — `\comment{}` / `\reply{}` / `\changes{}` macros, latexdiff workflow, writing-style rules. Read that file before touching response letter files.

When the super-project's pin lags behind the submodule's actual checked-out commit, that's normal during active work. Only bump the pin (`git add reports && git commit`) when you want the submodule's state recorded for posterity.

## Directory map (things worth knowing)

```
cycling_safety_perception/
├── PLAN.md                             # immediate handoff plan (read first)
├── CLAUDE.md                           # this file
├── pyproject.toml / uv.lock            # Python deps (use `uv sync`)
├── .venv/                              # uv-managed Python 3.10.16 env (gitignored)
├── data/                               # NOT in git; rsync separately
│   ├── raw/
│   └── processed/
├── logs/                               # modeling script logs (gitignored)
├── scripts/
│   └── run_demographics_with_retry.sh  # biogeme-silent-exit-resistant wrapper
├── cycling_safety_svi/                 # main package
│   ├── modeling/                       # all model scripts (see table above)
│   ├── perception_models/               # deep-learning perception model (separate pipeline)
│   └── ...
└── reports/                            # Overleaf submodule (main.tex etc.)
    ├── main.tex
    ├── models/                         # auto-generated \input{} tables
    │   ├── mxl_choice_<ts>/
    │   ├── stepwise_train_test_<ts>/
    │   └── interaction/safety_demographics_<ts>/<group>/
    └── CLAUDE.md                       # Overleaf + response-letter rules
```

## Git remotes

- Super-project: `git@github.com:koito19960406/cycling_safety_perception.git` (branch `main`)
- Reports submodule: `git@github.com:koito19960406/cycling_safety_perception_report.git` (branch `main`)

## Cloning onto a new machine

Clone the super-project **with** the `reports/` submodule populated, then install the uv env. Data files must be transferred separately (see *Data is not in git*).

```bash
# 1. Clone including the reports submodule.
git clone --recurse-submodules git@github.com:koito19960406/cycling_safety_perception.git
cd cycling_safety_perception

# If you already cloned without --recurse-submodules, run these instead:
#   git submodule update --init --recursive

# Confirm the submodule is populated on its `main` branch (not detached).
(cd reports && git status && git branch -vv)
#   expected: "On branch main", tracking origin/main

# 2. Install the uv-managed Python env.
curl -LsSf https://astral.sh/uv/install.sh | sh   # skip if uv is already installed
uv sync                                            # creates .venv/ from pyproject.toml + uv.lock

# 3. Sanity-check imports.
.venv/bin/python -c "import biogeme, pandas, numpy; print('ok')"

# 4. Copy data/ from the other machine (it's gitignored).
#    From the source machine:
#      tar czf /tmp/cycling_data.tgz data/
#      scp /tmp/cycling_data.tgz newhost:/path/to/cycling_safety_perception/
#    On the new machine:
#      tar xzf cycling_data.tgz && rm cycling_data.tgz

# 5. Confirm the cleaning assertion still passes with real data.
.venv/bin/python -c "
import pandas as pd
from cycling_safety_svi.modeling.mxl_functions import apply_data_cleaning
df = apply_data_cleaning(pd.read_csv('data/raw/cv_dcm.csv'))
print(df.shape, df.RID.nunique())"   # expect: (11190, 12) 746
```

If the submodule ends up in detached-HEAD state (common after `git submodule update --force`), re-attach it before editing:
```bash
cd reports && git checkout main && git pull --ff-only && cd ..
```

## Running state when you arrived (if relevant)

If the auto-retry wrapper is still going, `ps -ef | grep run_demographics_with_retry` will show the bash loop and its python child. The wrapper is safe to `pkill` — all completed groups persist on disk and the next invocation will resume from the same checkpoint directory. See `PLAN.md` Step 0 for the kill + push sequence.
