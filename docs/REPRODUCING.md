# Reproducing AutoPolish

Three tiers, cheapest first. Each tier is self-contained, so pick the deepest one your hardware
and data allow.

| Tier | Needs | Wall clock | Reproduces |
|---|---|---|---|
| [**1**](#tier-1--figures-from-cached-results) | CPU + experiment outputs (2.2 GB) | minutes | every figure and number in the paper |
| [**2**](#tier-2--re-analysis-from-raw-logs) | CPU + experiment outputs (2.2 GB) | ~1 hour | the analysis JSONs themselves, from the raw run logs |
| [**3**](#tier-3--full-runs-from-scratch) | GPUs + source datasets | days | the raw run logs, from scratch |

Every run is seeded (`--seed 0`), and every confidence interval in the paper comes from 1,000
bootstrap resamples of images or raters.

---

## Setup

```bash
git clone https://github.com/forall-labs/AutoPolish && cd AutoPolish
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU: analysis, figures, data download
```

Tiers 1 and 2 need nothing else. Tier 3 adds the GPU stack (see [below](#gpu-environment)).

---

## Tier 1 — figures from cached results

Download the experiment outputs from the Hugging Face Hub (public, no token needed):

```bash
python tools/fetch_from_hf.py autopolish
```

This restores `results/` (analysis JSONs and figures) and `data/results/` (raw run logs and
edited images). Then redraw every figure in the paper:

```bash
cd analysis
python paper_figs.py        # all main-text and appendix figures -> results/paper_figs/
python appendix_figs.py     # the calibration figure of the appendix
```

### Where each result lives

The script names carry the experiment labels we used during development. The paper presents
the same results in this order:

| Label | Result in the paper | Script | Output |
|---|---|---|---|
| Exp 0 | how much one rating shares with other raters, and how reliable a group average is | `exp0_ceiling.py` | `results/exp0.json` |
| B4 | calibration of the audience's ratings | `calibration.py` | `results/calibration.json` |
| | calibration carried over between datasets | `calib_transfer.py` | `results/calib_transfer.json` |
| B1 | the persona moves the VLM's scores the way real groups differ | `steerability.py` | `results/steerability.json` |
| | the comments reveal the viewer's profile | `rationale.py` | `results/rationale.json` |
| C1 | separation between viewer groups, with Holm correction | `c1_separation.py`, `holm.py` | `results/c1_separation.json`, `results/holm.json` |
| C3 | agreement with the crowd on AI-generated image pairs, by panel size | `c3_rapidata.py` | `results/c3.json` |
| C2 | panel size on real images | `c2_ncurve.py` | `results/c2.json` |
| C4 | AutoPolish against the other critics over ten rounds | `c4_trajectory.py` | `data/results/c4_run2/analysis/c4.json` |
| C4 | image-by-image comparison of the critics | `c4_paired.py` | `results/c4_paired.json` |
| C4 | ablations of the acceptance rule, similarity floor and candidates | `c4_ablation.py` | `results/c4_ablation.json` |
| | robustness to sampling temperature | `temp_compare.py` | `results/temp_compare.json` |
| | checks against memorization | `leakage.py`, `repeated_measures.py` | `results/leakage.json`, `results/repeated_measures.json` |
| | every rated aspect, and content categories | `dims_extended.py`, `content_category.py` | `results/dims_extended.json`, `results/content_category.json` |
| | fairness across rater attributes | `bias.py` | `results/bias.json` |
| | the earlier self-refine loop | `c4_selfrefine.py`, `c4_selfrefine_rescore.py` | `results/c4_selfrefine.json` |

`c4_run2` is the editing run reported in the paper. `c4_run1` is an earlier run kept for
comparison.

---

## Tier 2 — re-analysis from raw logs

Everything in `results/*.json` can be re-derived from `data/results/` without a GPU. This is
the tier that checks the analysis rather than trusting the cache.

```bash
cd analysis

# the headline analyses
python exp0_ceiling.py
python calibration.py
python c1_separation.py
python c3_rapidata.py

# everything
for f in *.py; do
  case "$f" in common.py|theme.py|attrs.py|paper_figs.py|appendix_figs.py|paper_extra.py) continue;; esac
  echo "== $f"; python "$f" || echo "FAILED $f"
done
```

Each script writes `results/<name>.json`, plus figures under `results/figs/`. They should match
the downloaded copies up to float formatting.

The editing analyses read a run tree rather than `data/results/` directly, so they take
`--output-root`:

```bash
python c4_trajectory.py  --output-root ../data/results/c4_run2
python c4_qualitative.py --output-root ../data/results/c4_run2
python c4_progression.py --output-root ../data/results/c4_run2
```

---

## Tier 3 — full runs from scratch

### Source data

Download PARA, EVA, LAPIS and Rapidata from their official sources and lay them out under
`data/` as described in [DATASETS.md](DATASETS.md).

### GPU environment

```bash
tools/setup_gpu.sh            # venv, torch, requirements-gpu.txt, CUDA and Hugging Face checks
```

Models used: `Qwen/Qwen2-VL-7B-Instruct` (the audience and the critic),
`black-forest-labs/FLUX.1-Kontext-dev` (the editor), the LAION aesthetic predictor (CLIP ViT-L/14
with an MLP head, the held-out scorer) and `facebook/dinov2-base` (the similarity check).
FLUX.1-Kontext-dev is gated, so accept its license on the Hub and log in with
`huggingface-cli login` first.

### The synthetic audience on real raters

These runs produce `data/results/{para,eva,lapis}_{full,blind}`, the input to tiers 1 and 2.

```bash
python experiments/para_pipeline.py  --n-images 2000 --seed 0
python experiments/eva_pipeline.py   --n-images 4070 --seed 0
python experiments/lapis_pipeline.py --n-images 4000 --seed 0

# the persona-blind control
python experiments/para_pipeline.py --n-images 2000 --seed 0 --persona-blind --temperature 0.7
```

> **Pair `--persona-blind` with `--temperature > 0`.** Under greedy decoding every rater of an
> image gets an identical answer, so the per-image distribution collapses to a single value.

The sampled arms (`*_full_t07`) are the same commands with `--temperature 0.7`, with PARA
extended to `--n-images 4000`.
`--analyze-only <log.json>` recomputes every metric from an existing log without loading a
model. Per-dataset details on sharding, resuming and growing a run are in
[`docs/datasets/`](datasets/).

### AI-generated image pairs

`experiments/rapidata_pipeline.py` replays real preference voters, each described by the only
attributes Rapidata records (country and language), choosing between two AI-generated images.

```bash
# four GPUs over disjoint pairs, then merge the shard logs
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python experiments/rapidata_pipeline.py \
    --n-pairs 800 --shard $i/4 --temperature 0.7 \
    --output data/logs/rapidata_run.json & done; wait

python experiments/rapidata_pipeline.py --analyze-only \
    "$(ls -m data/logs/rapidata_run.shard*of4.json | tr -d ' \n')"

# the persona-blind control
python experiments/rapidata_pipeline.py --n-pairs 800 --persona-blind --temperature 0.7
```

The default `--criterion preference` withholds the generation prompt, because the human voters
did not see it either. A persona here is a group (country and language), since the data has no
stable rater ids.

### The AutoPolish editing loop

```bash
OUTPUT_ROOT=outputs/my_run tools/run_autopolish.sh
```

The defaults match the paper: 10 rounds, 100 photographs (50 PARA, 50 EVA), four critics
(`static`, `blind`, `society`, `reward_only`), 3 candidates per round, the FLUX editor, a
similarity floor of 0.78 and a panel of 10 personas. The script shards the images across all
visible GPUs, resumes after interruptions, and runs the analysis at the end. The step-by-step
runbook, with a smoke test, is in [RUNNING_ON_GPU.md](RUNNING_ON_GPU.md).

The `society` critic is AutoPolish. The other three are the baselines of the paper: a fixed
"improve this image" instruction, a single persona-free critique, and a scorer-guided reference
that optimizes the aesthetic predictor directly.

### The self-refine comparison

`experiments/claim4_pipeline.py` is the earlier, simpler loop the paper compares against: one
fixed persona per image, edits chained on the previous output rather than the original, and the
critic also acting as the scorer.

```bash
python experiments/claim4_pipeline.py --n-images 100 --n-iterations 10 --seed 0
```

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Fetch fills the disk | The snapshot is cached and then copied, so budget about twice the download. `huggingface-cli delete-cache` afterwards. |
| `no summary shards for run <name>` | `data/results/<run>/` is missing. Run `python tools/fetch_from_hf.py autopolish`. |
| Every persona returns the same score | Greedy decoding collapse. Pass `--temperature 0.7`. |
| FLUX download fails with 403 | FLUX.1-Kontext-dev is gated. Accept its license on the Hub with the account you are logged in with. |
| Out of memory in the editing loop | Pass `--cpu-offload`, or reduce `--candidates`. The full loop needs about 45 GB of GPU memory. |
| Figures look tiny when scaled | `paper_figs.py` draws at the final printed width (5.5 in). Include its output at that width. |
