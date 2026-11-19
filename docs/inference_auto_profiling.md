# Transfer-Auto step profiling and cross-view ablation

Run from the Framework root in the environment from [Auto inference](inference_auto.md).
The command-line runner prepares the same seven-camera daytime example as the native
notebook, downloads/reuses the checkpoint, launches native inference, and checks the
generated camera videos and optional numeric LiDAR. It needs no Jupyter server.

```bash
export COSMOS_INTERNAL=0 COSMOS_TRAINING=0

# Reference attention from the checkpoint.
./.venv/bin/python -m cosmos_framework.scripts.run_transfer_auto \
  --mode camera --frames 61 --resolution 480 --seed 0 \
  --warmup 0 --num-iterations 1 \
  --profile-steps --profile-step-attention \
  --output outputs/auto_attention_baseline

# Identical inputs and sampler; remove cross-view generation attention.
./.venv/bin/python -m cosmos_framework.scripts.run_transfer_auto \
  --mode camera --frames 61 --resolution 480 --seed 0 \
  --warmup 0 --num-iterations 1 \
  --profile-steps --profile-step-attention \
  --disable-cross-view-attention \
  --output outputs/auto_attention_same_view
```

Both commands use all visible GPUs by default. Set `--gpus N` to choose a smaller
allocation; context parallelism follows that count. Memory fit still depends on
the hardware and request. Output directories must be new. The supplied sample
uses the initial 61 frames at 30 FPS and seven physical camera identities.
To reuse inputs/weights explicitly, pass `--sample-root /path/to/sample_data`
and `--checkpoint /path/to/Cosmos3-Nano-Transfer-Auto`. A custom single-sample
JSON/JSONL request uses `--request /path/to/input.jsonl`; its camera mapping,
caption paths and frame interval are preserved. See `--help` for sampling knobs.
`--prepare-only` checks/prepares inputs without downloading model weights or
launching inference. `--mode text2video` omits initial RGB; `--mode joint` includes
LiDAR. To generate a shorter initial interval without supplying a request file,
add `--frames 61` (or another frame count of at least 5 on the `4k+1` grid).
The runner sets both the request and chunk frame counts, writes caption copies
covering `[0, N)` under the new run directory, and clips timestamped descriptions
using the shared caption-window helper. Original captions and videos are unchanged;
the inference loader reads the selected control interval. Requested frames must
fit the available controls and one initial caption interval. The bundled sample
supports up to 297 aligned frames. See the
[request guide](../.agents/skills/cosmos3-nano-transfer-auto/inference/references/input-and-execution.md#caption-sidecars)
for preparing other intervals and longer continuation inputs.

## Timing boundaries and reports

`--profile-steps` also works directly with `cosmos_framework.scripts.inference`.
It surrounds each complete solver step: the conditional/unconditional model
evaluations, CFG combination, and scheduler update. These boundaries follow the
diffusion-timestep timing in the BCA implementation. Loading, input encoding,
final latent decoding and file export are outside the step interval.
UniPC, fixed-step, EDM (including multi-evaluation solvers), and component UniPC
use the same instrumentation. A separate `sampler_run` identifies each new
solver loop, such as a continuation chunk; step indices restart from zero.

Each sample directory contains files such as:

```text
diffusion_steps.batch000.measured000.rank0000.csv
```

Only global rank 0 records and writes the profiling CSV. Other ranks create no
step timers or profiling files; no timing JSON or Chrome trace is produced by
`--profile-steps`. The commands above each perform exactly one generation.
Rows include `sampler_run`, zero-based `step_index`, total step count, timestep,
status, synchronized `wall_ms`, and current-stream `cuda_elapsed_ms`.
The timestep uses the sampler's native units (sigma for EDM); component UniPC
records the first component's timestep. Rank, phase and iteration are CSV columns.
Warmup runs remain separate from measured iterations; none of the first steps
is silently dropped. Failed steps are marked and partial reports are retained.
The measurements describe rank 0's device and host; they are not a reduction
over all GPUs.

`--profile-step-attention` adds disjoint region totals and call counts for
`same_view`, `cross_view`, `caption`, and `attention_merge`, summed over layers
and CFG evaluations in each step. It requires `--profile-steps` and eager
maskless `decomposed` or `same_view` attention. The runner already disables
compilation and CUDA graphs; direct CLI users must pass
`--no-use-torch-compile --no-use-cuda-graphs`. This flag does not silently
replace a checkpoint's FlexAttention backend. Other model/packing/communication
work remains in the step total and is not attributed to these four regions.

CUDA events are read after the step's device synchronization; individual
attention regions never synchronize the GPU. An event interval measures
current-stream elapsed time, including possible dispatch gaps, rather than
summed kernel execution time. Profiling synchronization and region events add
overhead: use identical flags for comparisons. Use total-only `--profile-steps`
for a less intrusive timing run and separate detail runs to explain the result.

## No-cross-view experiment

`--disable-cross-view-attention` applies model-construction overrides to the
in-memory configuration before the network is built. The saved checkpoint is
never edited. `multiview_attention.mask.attention_scope` becomes `same_view`,
and cross-view windows, band expansion, action-crossing flags and cross-view
pose transforms are disabled. Within-view temporal attention, configured
same-view control connections, caption access and learned camera identities
remain. The maskless backend builds no cross-view partitions and launches no
cross-view kernels; enabled detailed timing therefore reports zero cross-view
calls. FlexAttention also honors the same-view scope, but region breakdowns
are specific to maskless attention.

Cosmos3 merges same-view generation keys with caption keys in unified attention,
so that merge remains in this ablation. It is not a cross-view operation. This
differs from BCA's separate self/cross-view operator, which can return its self
partition directly when all cross-view sources are removed.

For joint generation this removes cross-camera and camera/LiDAR GEN-to-GEN
connections. Caption conditioning retains the checkpoint's policy. This is a
no-cross-view ablation, not BCA's geometry/global/residual token-selection method.
The flag requires a multiview request and rejects the causal Sim-Auto runtime,
whose cached attention is a separate implementation. Check image quality and
cross-view consistency alongside timing: the ablation intentionally removes
the mechanism that exchanges observations between cameras.
