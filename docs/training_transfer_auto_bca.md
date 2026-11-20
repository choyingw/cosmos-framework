# Transfer-Auto BCA teacher–student dummy training

This extends the [dense edge adaptation](training_transfer_auto_edge.md) to BCA
using the same daytime sample as `python -m cosmos_framework.scripts.run_transfer_auto`:
seven synchronized views, first **61 frames**, **480 × 832**, camera-local captions
and online Canny edge controls. The default run performs **10 optimizer updates**.

The teacher must be your **previously edge-finetuned, exported dense model**.
The launcher validates its edge-task metadata and converts its exact weights into
a separate cached DCP. An initialization path left over from the original WSM run
cannot replace an explicitly supplied teacher checkpoint. Regular teacher weights
load strictly; new LoRA/register parameters initialize separately, and the optimizer
and student EMA start fresh. VAEs and tokenizer resolve automatically.

## Paper and Bosch settings

The implementation follows sections 3.2–3.5 and 4.3 of the supplied
`5919_Budgeted_Cross_View_Atten.pdf`, and the sibling Bosch code:

- `Bosch_GenWM/cosmos_transfer2/_src/transfer2_multiview/utils/cross_view_budget_attn.py`:
  F-theta/pinhole camera projection, inverse-depth local neighborhoods, shared
  register queries, and per-head residual selection.
- `Bosch_GenWM/cosmos_transfer2/_src/transfer2_multiview/models/multiview_vid2vid_model_control_vace_rectified_flow.py`:
  dense teacher with adapters disabled and no gradients; budget student on the
  same noisy input and timestep; prediction and pooled-feature distillation.
- `Bosch_GenWM/cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/experiment/waymo.py`:
  rank/alpha 16, blocks 10–20 inclusive, local/global budgets 32/512,
  prediction/feature weights 1.0/0.5, feature pooling 4, and no training residual bank.

| Component | Setting |
| --- | --- |
| Trainable student parameters | Generation Q/K/V/O and MLP rank-16 LoRA, alpha 16; learned register queries |
| Frozen teacher | Dense edge weights shared with the student; LoRA and BCA disabled during teacher forward |
| BCA blocks | Zero-based blocks 10–20 inclusive; other blocks retain native dense cross-view attention |
| Local bank | At most 32 same-frame tokens across neighboring cameras and control/target source roles |
| Local geometry | 8 log-spaced depths, 1–120 m; 128 inverse-depth-weighted candidates per source before deduplication/capping |
| Global bank | 512 learned summaries, spatial average pooling kernel/stride 8; neighboring cameras |
| Training residual bank | Off |
| Inference residual bank | 35% of eligible tokens per other source camera, head and control/target role |
| Residual importance | Protected random representative per 2×2×2 cell, then largest squared V-feature differences; same indices for K/V; no merging |
| Loss | Native flow matching + prediction MSE + 0.5 × spatially pooled block-feature MSE |
| Optimizer | Existing native AdamW, learning rate 1e-5; base weights frozen |

Cosmos 3 has shared generation experts rather than a separate control branch.
BCA therefore processes both control and target tokens in those shared blocks.
The local budget remains **32 total**, split between source roles. Local, global,
residual, same-view and caption partitions use a **joint softmax normalizer**.
Feature loss supervises target-camera features, pooled spatially by 4 without
pooling time. Teacher and student see identical sampled noise, timestep and
conditioning at every update.
Feature hooks retain output references only; spatial pooling and CP gathering run
after the forward returns, outside activation-checkpointed blocks. This preserves
feature gradients while keeping forward/recomputation tensor counts identical
after the temporary hooks are removed before backward.
The fresh student run enables native CP gradient correction for both prediction
and feature gathers, preserving their relative loss weights across GPU counts.

This is an adaptation to Cosmos 3's native attention layout: same-view attention
stays dense and cross-view visibility keeps its **preceding 0.4-second window**.
It does not change the native teacher into Bosch's unrestricted full-4D teacher.
CUDA FP16/BF16 now uses a fused implementation adapted from Bosch's indexed
attention and reusable source-summary strategy:

- Pool each source frame once and reuse it across overlapping time windows.
- Batch all camera views and both query roles for FlashAttention register and
  shared-bank attention; keep grouped-query heads without expanding K/V heads.
- Load local K/V directly from calibrated indices in Triton, with four queries
  per block for the 32-token budget. Inference fuses the local/shared LSE merge.
- Batch residual importance scoring in Triton, caching only fixed random
  representatives and geometry routes. Input-dependent scores, selected tokens,
  K/V, and learned summaries are recomputed every forward.
- Preserve output **and LSE gradients** through FlashAttention and indexed
  attention, including native partition merges and activation checkpointing.

The parameter names and shapes are unchanged; existing exports work without
retraining or re-exporting. The ordinary training/inference commands select this
path automatically. `COSMOS_BCA_BACKEND=reference` restores the previous chunked
PyTorch implementation for diagnosis. CPU/unsupported dtypes also use that
reference path. The CUDA path uses PyTorch's built-in FlashAttention operator
and Triton; it does not require installing Bosch's package.

The same step profiler measures the entire cross-view operation, including
geometry lookup, compression and selection, and still writes only rank 0 CSV.
The first call may include Triton compilation and static-cache construction.

### Isolated operator measurements

Measured on an RTX 3090, PyTorch 2.8.0+cu128 / Triton 3.4.0, BF16, seven views,
both control/target roles, 32 query / 8 K/V heads, head dimension 128, local/global
budgets 32/512. Each number is one measured call after one untimed warmup.
Inference uses 16×15×26 tokens per view/role, corresponding to 61 frames at
832×480 with temporal compression 4 and spatial compression/patchification 32.
Training uses **two latent frames** to fit the reference's retained activations.

| Operator workload | Reference CUDA time | Fused CUDA time | Reference extra peak allocation | Fused extra peak allocation |
| --- | ---: | ---: | ---: | ---: |
| Inference, 16 latent frames, 35% residuals | 3899 ms | 281 ms | 1513 MiB | 1403 MiB |
| Training forward + backward, 2 latent frames, no residuals | 531 ms | 67 ms | 12705 MiB | 1584 MiB |

These are BCA-block measurements, **not complete diffusion-step or generation
speedups**. Allocations exclude live inputs/parameters and warmed static caches;
they include the operator's outputs and temporaries. Inference sampled-output
relative L2 difference was 0.00315, consistent with BF16 reduction differences.
They are not bitwise-identical implementations. Training uses FP32 atomic
accumulation for local K/V gradients, so backward is not bitwise deterministic.

To reproduce a single measured comparison without loading model weights or
generating videos (CSV to stdout, rank 0 only):

```bash
python -m cosmos_framework.scripts.benchmark_bca
python -m cosmos_framework.scripts.benchmark_bca --mode training
```

The bundled reference calibration is copied from `Bosch_GenWM/calibration.json`.
Its match to the default daytime sample is **assumed at the user's instruction**.
Pass `--bca-calibration /path/to/calibration.json` to replace it. The launcher embeds
the selected JSON in `train.env` and model config, so an exported model needs no
external calibration file. The inference wrapper validates the seven camera
identities and orders complete view records to match the exported calibration
before preparing edge maps and captions. Source request order may differ; RGB,
controls, captions and prompts remain paired with their camera key.

## Train ten updates

Run in the existing Framework environment on the training machine. Set `EDGE_HF`
to the successful dense edge export from the preceding experiment. Use a new
output directory for each dummy run.

```bash
cd /workspace/cosmos-framework
source .venv/bin/activate
export BCA_RUN=/workspace/cosmos_data/bca_dummy_run01

python -m cosmos_framework.scripts.prepare_transfer_auto_edge_sample \
  --train-dummy --bca --steps 10 \
  --checkpoint "${EDGE_HF:?Set EDGE_HF to your exported dense edge teacher}" \
  --output "$BCA_RUN"
```

This reuses the default inference sample downloader/cache. For a sample downloaded
outside that cache, add `--sample-root /path/to/assets/sample_data`. GPU count
defaults to all visible GPUs; add `--gpus N` to select it. There is one frozen
teacher forward and one student forward per optimizer update, with no generated
videos or validation inference during this dummy training.

The launcher saves `train.env`, `command.txt`, a data bundle, and the native
training directory at `$BCA_RUN/training/cosmos3/transfer_auto/bca_dummy`.
The checkpoint is saved at update 10.

## Export the trained student

Export the DCP iteration directory directly; no `.pt` conversion is needed.
The native export bundles base weights, student LoRA weights, register weights,
calibration/config and tokenizer, plus the LiDAR VAE required by the architecture.
`--no-vit` is required for this generation-only recipe.

```bash
source "$BCA_RUN/train.env"
export BCA_TRAIN="$BCA_RUN/training/cosmos3/transfer_auto/bca_dummy"
export BCA_HF="$BCA_RUN/bca_hf"

COSMOS_TRAINING=1 python -m cosmos_framework.scripts.export_model \
  --config-file "$BCA_TRAIN/config.yaml" \
  --checkpoint-path "$BCA_TRAIN/checkpoints/iter_000000010" \
  --output-dir "$BCA_HF" \
  --no-vit --bundle-lidar-vae \
  --per-view-captions on \
  --per-view-caption-task av_multiview_edge_transfer
```

The exporter uses the student EMA by default. It retains the LoRA modules and
BCA architecture; no separate adapter merge is required for Framework inference.

## One inference with the existing profile

```bash
python -m cosmos_framework.scripts.run_transfer_auto \
  --checkpoint "$BCA_HF" \
  --control-type edge --mode camera \
  --frames 61 --resolution 480 --seed 0 \
  --warmup 0 --num-iterations 1 \
  --profile-steps --profile-step-attention \
  --output "$BCA_RUN/inference"
```

Inference activates the exported **35% residual retention** automatically, in
addition to the 32 local candidates and 512 summaries. That rate applies to
selected blocks' eligible cross-view tokens, rather than 35% of every model token.
This command performs one generation, with no warmup or repeated benchmark.

Edge maps are saved under `$BCA_RUN/inference/edge_controls/view_00.mp4` through
`view_06.mp4`; `request.json` records camera mappings. Generated camera videos
and the profile are under `$BCA_RUN/inference/generated/`. The existing profiler
writes **only rank 0 CSV**, named
`diffusion_steps.batch000.measured000.rank0000.csv` in the sample output directory.
The columns remain `wall_ms`, `cuda_elapsed_ms`, same-view/cross-view/caption/merge
dispatch and CUDA times and call counts. No trace or JSON timing report is enabled.

## Validation limits

CPU tests exercise the native maskless dispatch, joint-softmax backward, bounded
geometry routes, temporal visibility, importance selection, config round trips,
and ten optimizer updates through the actual teacher–student helper on a tiny
network with checkpointing off, full, and selective. Feature-only gradients also
match the uncheckpointed reference. The frozen teacher remains unchanged while LoRA/register parameters
update with both the reference and batched implementations. A native training-config
dry run resolves the ten-iteration launch. CUDA regression tests compare fused
outputs and Q/K/V/register gradients against the reference, including FP16/BF16,
grouped-query attention, LSE-only gradients, empty routes, sequence padding,
importance selection, future-frame exclusion, and full/selective checkpointing.
Full-size GPU training, checkpoint export/readback and generation require the
model assets and training GPUs; these have not been executed locally.

## Save error: cannot pickle code objects

On older PyTorch with Python 3.13, DCP can fail to serialize the traceback of an
earlier checkpoint-write error. This hides the actual exception behind
`TypeError: cannot pickle code objects` during `gather_object`. See
[PyTorch issue 174669](https://github.com/pytorch/pytorch/issues/174669).
The Framework save worker now scopes the upstream traceback cleanup to each
DCP save, retaining the original exception and reporting the failing component
and destination on every affected rank. Checkpoint contents are unchanged;
failed saves do not publish `latest_checkpoint.txt`.

That compatibility fix exposes the original write error; it does not establish
why writing failed. Before repeating training, check free space and inodes on the
actual output filesystem:

```bash
df -h /workspace
df -i /workspace
du -sh "$BCA_RUN/training/cosmos3/transfer_auto/bca_dummy/checkpoints"
```

The checkpoint includes the full base model and student EMA even though only
LoRA/register parameters train. A failed save may leave partial shard files.
Use the original exception to identify the cause before deciding what to retry
or remove; an incomplete iteration directory is not a resumable checkpoint.
