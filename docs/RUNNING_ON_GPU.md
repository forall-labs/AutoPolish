# Running the editing experiment on a GPU node

A copy-paste runbook for the full AutoPolish experiment: set up the environment, run a smoke
test, launch the full run, and read its outputs. Everything is plain scripts, no notebook.

## Requirements

- **GPU.** One H100 or A100 per shard. The editor (FLUX.1-Kontext), the audience
  (Qwen2-VL-7B), the aesthetic scorer (CLIP) and the similarity check (DINOv2) together need
  about 45 GB, so one 80 GB GPU per shard is comfortable.
- **Hugging Face login** with the license of
  [`black-forest-labs/FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
  accepted on its model page. Without it the editor download fails with 403.
- **PARA and EVA** under `data/`, laid out as in [DATASETS.md](DATASETS.md).

## 1. Environment

```bash
huggingface-cli login
tools/setup_gpu.sh
source .venv/bin/activate
```

`setup_gpu.sh` creates `.venv`, installs torch and `requirements-gpu.txt`, prints the visible
GPUs, checks that `diffusers` and `editor` import, and checks your Hugging Face login.

On a cluster:

- torch from a module: `module load <cuda/torch>`, then `SKIP_TORCH=1 tools/setup_gpu.sh`
- a specific CUDA wheel: `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 tools/setup_gpu.sh`
- the venv elsewhere: `VENV=/path/to/venv tools/setup_gpu.sh`

## 2. What the experiment does

For each image and for 10 rounds, a **critic** looks at the current best image and its
complaints are condensed into one edit instruction. **FLUX.1-Kontext** applies the accumulated
instruction to the **original** image, producing K candidates. Each candidate is scored by the
**LAION aesthetic predictor**, a different model family from the critic, and checked for
**identity drift** with DINOv2. A candidate replaces the current best only if its score improves
and its similarity to the original stays above the floor, so the best-so-far score can only rise.

Four critics are compared:

| Condition | Critic |
|---|---|
| `static` | a fixed "improve this image" instruction |
| `blind` | one persona-free VLM critique |
| `society` | **AutoPolish**: a panel of 10 personas, aggregated |
| `reward_only` | a reference that optimizes the aesthetic predictor directly |

Defaults (already set, no flags needed): 10 rounds, 100 images (50 PARA and 50 EVA, low to mid
rated), K = 3 candidates, the FLUX editor with guidance 3.0 and a "make this a clearly visible
edit" suffix on every instruction, and a similarity floor of 0.78.

## 3. Smoke test (2 images, ~2 minutes)

Loads every model and writes a few edits, so any failure shows up immediately:

```bash
python experiments/c4_refine.py --dataset eva --n-images 2 \
    --conditions static,blind,society --editor flux --steps 3 --candidates 2 \
    --output-root outputs/smoke
```

Then run the analysis on it, to check the whole path from loop to figures:

```bash
cd analysis
python c4_trajectory.py  --output-root ../outputs/smoke
python c4_qualitative.py --output-root ../outputs/smoke
```

`outputs/smoke/analysis/` should hold `c4.json`, `c4_summary.md` and six figures. The numbers
mean nothing at this size; the point is that everything runs.

## 4. Full run

```bash
OUTPUT_ROOT=outputs/my_run tools/run_autopolish.sh
```

The script shards the images across all visible GPUs (one shard per GPU), resumes safely if
re-run after an interruption, and builds the figures and the results table at the end. For a
long run, detach it:

```bash
nohup env OUTPUT_ROOT=outputs/my_run tools/run_autopolish.sh > my_run.out 2>&1 &
tail -f my_run.out
```

Optional overrides:

```bash
NGPU=4 OUTPUT_ROOT=… tools/run_autopolish.sh                        # force a 4-way shard
TOTAL_IMAGES=200 OUTPUT_ROOT=… tools/run_autopolish.sh              # more images
CONDITIONS=static,blind,society OUTPUT_ROOT=… tools/run_autopolish.sh
EXTRA_ARGS="--drift-cap 0.80" OUTPUT_ROOT=… tools/run_autopolish.sh # stricter similarity floor
RUN_ANALYSIS=0 OUTPUT_ROOT=… tools/run_autopolish.sh                # skip the analysis step
```

**Compute.** 100 images × 4 critics × 10 rounds × 3 candidates is about 12,000 FLUX edits plus
the persona critiques: a few hours on one H100, and proportionally less across several GPUs.

## 5. Outputs

```
$OUTPUT_ROOT/
  edits/<condition>/<image_id>/
      step0_source.png, step*_cand*.png, step*_best.png   # source, every candidate, committed best
  logs/c4_<condition>/
      c4_<condition>*.json                                # run summary (config, panel, floor)
      c4_<condition>*.part-*.json                         # per-round records
  analysis/
      c4.json            # every metric: gains, win rates, convergence, drift, diversity
      c4_summary.md      # the results table
      figs/              # trajectory, raw objective, headline, drift, diversity, qualitative grid
  stdout/                # per-GPU logs
```

To redraw the figures later, for example after copying the folder elsewhere:

```bash
cd analysis
python c4_trajectory.py  --output-root ../outputs/my_run
python c4_qualitative.py --output-root ../outputs/my_run
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| 403 when loading FLUX | Accept the FLUX.1-Kontext-dev license on the Hub and log in with that account. |
| CUDA out of memory | One 80 GB GPU per shard, lower `--candidates`, or `EXTRA_ARGS=--cpu-offload` (slower). |
| torch cannot see CUDA | `TORCH_INDEX_URL=…/cuXXX tools/setup_gpu.sh`, or `SKIP_TORCH=1` after `module load`. |
| The run died partway | Re-run the same command. It resumes and skips finished images. |
| One shard failed | Check `$OUTPUT_ROOT/stdout/c4_gpu*.log`. Re-running redoes only the missing work. |
| Edits are too subtle | `EXTRA_ARGS="--drift-cap 0.72"` lets bolder edits through. |
| Edits change the scene | `EXTRA_ARGS="--drift-cap 0.82"` keeps them closer to the original. |
| `No c4 logs found` from the analysis | Point `--output-root` at the folder that contains `logs/c4_*`. |
