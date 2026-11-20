# Seven-view edge adaptation: 61 frames, 480p

This recipe adapts **Cosmos3-Nano-Transfer-Auto** from WSM to Canny edge
controls. It uses all seven reference cameras together, exactly 61 frames per
camera at **480 × 832**, camera-local captions and the released cross-view
attention policy. The frame rate comes from the synchronized recordings; the
reference sample is 30 FPS, so 61 frames cover about two seconds. There is no
LiDAR task and no WSM input in this adaptation.

## What Bosch trained

In the sibling `Bosch_GenWM` checkout:

- `cosmos_transfer2/experiments/multiview/cosmos_multiview_example.py:93`
  defines `tfx105_transfer2_auto_multiview_post_train_hdmap2edge`. It inherits
  the released HD-map multiview experiment/checkpoint and changes `hint_keys`
  to `edge`. It does not enable LoRA or unfreeze the base model.
- `cosmos_transfer2/_src/transfer2_multiview/models/multiview_vid2vid_model_control_vace_rectified_flow.py:92`
  defaults `train_base_model=False`; its build path calls `freeze_base_model`.
- `cosmos_transfer2/_src/transfer2/models/vid2vid_model_control_vace_rectified_flow.py:467`
  freezes the network, then unfreezes the **full control blocks, control
  embedder and applicable control-specific projections/embedders/hint block**.
- The later
  `cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/experiment/factorized.py:16`
  explicitly selects `use_lora=True, unfreeze_params="lora_only"`. That is a
  separate factorized-attention adaptation.

Thus the original HD-map-to-edge adaptation is **dense control-branch
fine-tuning with a frozen base**, rather than LoRA or full-network fine-tuning.
This finding describes the checked-in configuration; it does not establish
which overrides a historical external training job used.

Cosmos 3 packs control and target as two vision items processed by shared MoT
generation experts, rather than exposing Bosch's separate control branch.
The new recipe therefore applies the corresponding **dense generation-side
adaptation**: `moe_gen`, `time_embedder`, `vae2llm`, `llm2vae` and
`rig_view_embed` are trainable; the language backbone, VAEs and LiDAR adapters
are frozen. LoRA is disabled. These experts also process target tokens, so this
is an architectural approximation of Bosch's parameter-freezing policy.

`MultiviewEdgeControl` reproduces Bosch's RGB Canny operation after the common
frame selection and resize/crop. Training samples lower thresholds in `[20,100)`
and a threshold difference in `[50,150)`, once per synchronized sample.
Inference uses Bosch's medium setting, **100/200**, with no extra grayscale
conversion or blur. Inference controls are stored as lossless RGB H.264 MP4 files.

## Prepare weights and recordings

Use your installed Framework training environment with working TorchCodec,
FFmpeg and OpenCV. Run commands from the Framework root; no notebook is needed.
Use a complete local release snapshot and the matching video VAE:

```bash
export COSMOS_TRAINING=1
export TRANSFER_AUTO_HF=/path/to/Cosmos3-Nano-Transfer-Auto
export COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP=/path/to/new/transfer_auto_init_dcp
export COSMOS_TRANSFER_AUTO_VIDEO_VAE=/path/to/Wan2.2_VAE.pth
export COSMOS_TRANSFER_AUTO_LIDAR_VAE="$TRANSFER_AUTO_HF/lidar_vae/diffusion_pytorch_model.safetensors"

python -m cosmos_framework.scripts.prepare_sim_auto_checkpoint \
  --checkpoint-path "$TRANSFER_AUTO_HF" \
  --output-path "$COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP"
```

The DCP contains the released generation weights and physical view embeddings.
The LiDAR tokenizer remains in the model architecture for checkpoint
compatibility and still needs its VAE file, although no LiDAR samples or loss
are used. Do not skip missing regular model weights to make a load succeed.

### Dummy training on run_transfer_auto's single sample

To repeat the **exact daytime source sequence** used when running
`python -m cosmos_framework.scripts.run_transfer_auto` without `--request`, run:

```bash
python -m cosmos_framework.scripts.prepare_transfer_auto_edge_sample \
  --train-dummy --steps 10 \
  --output /workspace/cosmos_data/edge_dummy_run01
```

This calls the **same `fetch_sample` helper** as the inference runner, reusing
its downloaded assets. It selects the first 61 frames from all seven recorded
RGB views, clips the supplied captions to that interval, computes Canny controls
online at 480x832, and repeatedly cycles that **one training sequence** for ten
optimizer updates. It launches training directly; no request file, manual index,
or `source train.env` step is needed. No model/VAE environment variables are
required for this command: the launcher resolves the released Transfer-Auto
snapshot and tokenizer from Hugging Face, uses its bundled LiDAR VAE, and
obtains the matched `Wan2.2_VAE.pth` from `Wan-AI/Wan2.2-TI2V-5B` if needed.
Existing cached files are reused.

The first launch converts the released pretrained weights to a CPU-created DCP,
then validates and caches it under
`$HF_HOME/cosmos_training/transfer_auto_init/` (the default Hugging Face home
is used when `HF_HOME` is unset). Subsequent launches reuse that conversion.
The cache identifies the source snapshot/config/weight files, and incomplete
conversions are not published for reuse. All resolved paths are also saved in
`train.env`. Missing gated-model access still requires your normal HF login.

Existing `COSMOS_TRANSFER_AUTO_EDGE_INIT_DCP`,
`COSMOS_TRANSFER_AUTO_VIDEO_VAE`, `COSMOS_TRANSFER_AUTO_LIDAR_VAE`,
`TRANSFER_AUTO_HF`, `LOCAL_QWEN_TOKENIZER` and `WAN_VAE_PATH` overrides remain
supported. To select a local pretrained snapshot directly, use
`--checkpoint /path/to/Cosmos3-Nano-Transfer-Auto`.

GPU count defaults to all visible GPUs; use `--gpus N` to choose a count. The
dummy run disables compilation, saves at its final update, and writes its
training outputs under `<output>/training` plus its command in `command.txt`.
It does not run generation or validation. Use a fresh output directory for each
dummy run.

### Prepare the same sample without starting training

Use the **recorded RGB** in `assets/sample_data/daytime_scene/rgb`, together
with its supplied captions. The preparation command uses the inference runner's
sample downloader with `AUTO_SAMPLE_ROOT` or its usual Hugging Face cache.
It creates the one-sequence edge bundle and writes the resolved paths to
`train.env`. Already downloaded assets are reused; missing sample files are fetched.

```bash
export EDGE_SAMPLE_TRAINING_DIR="$HOME/cosmos_data/edge_daytime_sample"
python -m cosmos_framework.scripts.prepare_transfer_auto_edge_sample \
  --tokenizer "$TRANSFER_AUTO_HF/text_tokenizer" \
  --output "$EDGE_SAMPLE_TRAINING_DIR"
source "$EDGE_SAMPLE_TRAINING_DIR/train.env"
```

For a download made with `hf download --local-dir`, add
`--sample-root /actual/download/directory/assets/sample_data`. Passing the
download root or `daytime_scene` also works. On Vast.ai this is the path on
that instance. No path to locally saved generated `vision_view*.mp4` output is
used for training.

This selects all seven cameras, preserves the downloaded captions and creates
clipped caption copies for the first 61 frames,
and removes the WSM/LiDAR requirements of the release's joint training index.
The recipe uses the same initial 61-frame interval as inference and
derives edges after resizing to 480x832. The sample has no held-out training
sequence. Continue directly to **Train and export** below; the custom-index
instructions that follow are for other recordings.

### Use other recordings

Build a JSONL index with one synchronized sequence per line. For each of the
seven cameras, provide `video_<camera_key>` and `caption_<camera_key>` assets.
Paths resolve relative to the index. A line has this structure, with **all
seven cameras** included in `assets`:

```json
{"id":"scene-0001","stratum":"480","assets":{"video_camera_front_wide_120fov":"scene-0001/rgb/camera_front_wide_120fov.mp4","caption_camera_front_wide_120fov":"scene-0001/captions/camera_front_wide_120fov.json"}}
```

Each caption file contains camera-local descriptions grounded in the recorded
RGB and a `caption_structured` object, for example:

```json
{"caption_structured":{"chunk_0_120":{"start_frame":0,"end_frame":120,"caption":"A white car travels along a dry road in daylight."}}}
```

Frame bounds are source-frame indices with an exclusive end. All camera videos
and caption windows must cover the same selected 61-frame interval at the same
FPS. Short windows are rejected, not silently shortened. Preserve supported
target captions; do not change training weather/appearance labels without
matching RGB targets. Split disjoint scenes/drives for training and validation.

```bash
export EDGE_INDEX=/path/to/train-index.jsonl
export COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE=/path/to/new/edge_bundle

python -m cosmos_framework.data.generator.local_simulation.auto_bundle \
  --index "$EDGE_INDEX" \
  --tokenizer "$TRANSFER_AUTO_HF/text_tokenizer" \
  --output "$COSMOS_TRANSFER_AUTO_LOCAL_BUNDLE" \
  --control-type edge \
  --camera-keys \
    camera_front_wide_120fov camera_cross_right_120fov \
    camera_rear_right_70fov camera_rear_tele_30fov \
    camera_rear_left_70fov camera_cross_left_120fov camera_front_tele_30fov
```

This copies RGB, captions and tokenizer files. Optional `--hardlink` avoids
copying media on the same filesystem; linked inputs must remain immutable.
The training loader derives edge maps online; no precomputed edge, WSM or
LiDAR dataset files are required. This is a distinct camera-only edge bundle,
not the original joint WSM/LiDAR bundle.

## Train and export

The experiment is `transfer_auto_edge_61_480`. Defaults are eight GPUs with
CP=8/FSDP=8, batch one sequence per replica, AdamW peak LR `1e-5`, 100 warmup
updates, 5,000 total updates and saves every 100 updates. These are starting
settings for the new adaptation, not a reproduction of Bosch's optimizer
schedule. Training mixes zero/one initial RGB conditioning frames 50/50.

```bash
export IMAGINAIRE_OUTPUT_ROOT=/path/to/training_outputs
export EDGE_RUN_NAME=edge_61_480_run01

entry=(
  --config cosmos_framework/configs/simulation/config.py
  --experiment-module cosmos_framework.configs.simulation.experiment.auto.transfer_auto_edge
)
settings=(experiment=transfer_auto_edge_61_480 "job.name=$EDGE_RUN_NAME")

python -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" --dryrun -- "${settings[@]}"

torchrun --standalone --nproc-per-node=8 \
  -m cosmos_framework.scripts.train_experiment \
  "${entry[@]}" -- "${settings[@]}"
```

For another allocation, change **all three** mesh settings together:
`model_parallel.context_parallel_size`,
`model.config.parallelism.context_parallel_shard_degree` and
`model.config.parallelism.data_parallel_shard_degree`. Seven-view dense training
needs substantially more memory than inference; validate a real forward/backward
and strict checkpoint readback on the chosen allocation.

The first load is weights-only and strict for regular parameters. A callback
copies the loaded regular model into a new EMA; optimizer, scheduler and
iteration counters start fresh. Same-job auto-resume restores its saved state
and preserves its EMA. Use a fresh job name for an independent adaptation.

The dummy-training launcher first checks that the initialization DCP exists and
contains regular generation-expert, timestep, video-adapter and rig-embedding
weights. Both VAE paths must exist. The native strict load then checks the full
model state and tensor shapes; absent model weights cannot silently turn the
dummy run into training from scratch. The source is the released checkpoint
converted by `prepare_sim_auto_checkpoint` above. No new control-specific branch
is initialized: the shared generation experts and adapters are adapted from
their pretrained values.

Export a completed training checkpoint using the **resolved config.yaml saved
by that run** (the dry run prints its location):

```bash
export EDGE_RUN_CONFIG=/path/to/the/run/config.yaml
export EDGE_TRAIN_CHECKPOINT=/path/to/the/run/checkpoints/iter_000005000
export EDGE_HF=/path/to/new/transfer_auto_edge_hf

python -m cosmos_framework.scripts.export_model \
  --config-file "$EDGE_RUN_CONFIG" \
  --checkpoint-path "$EDGE_TRAIN_CHECKPOINT" \
  --output-dir "$EDGE_HF" \
  --no-vit \
  --bundle-lidar-vae \
  --per-view-captions on \
  --per-view-caption-task av_multiview_edge_transfer
```

The recipe also lets export derive the edge task automatically. The exported
checkpoint records `per_view_captions: true` and
`per_view_caption_task: "av_multiview_edge_transfer"`; inference uses the same
edge-specific system instruction as training.

Use `--no-vit` for this generation-only workflow. The recipe clears
`vlm_config.pretrained_weights.backbone_path` because the released DCP supplies
the language/generation weights, and the exporter otherwise tries to resolve
that empty path for a reasoner vision tower. Camera RGB and Canny controls use
the video VAE; `--no-vit` keeps that tokenizer and the generation network intact.

## Generate without a request file

For the **original pretrained Transfer-Auto model**, run the following on the
same initial 61-frame daytime sequence. This uses its released WSM controls:

```bash
python -m cosmos_framework.scripts.run_transfer_auto \
  --checkpoint "$TRANSFER_AUTO_HF" \
  --control-type wsm --mode camera --frames 61 --resolution 480 --seed 0 \
  --warmup 0 --num-iterations 1 \
  --profile-steps --profile-step-attention \
  --output outputs/pretrained_sample_61
```

The profiling artifact is only
`diffusion_steps.batch000.measured000.rank0000.csv` under the generated sample
directory. It retains step wall/CUDA timing and same-view, cross-view, caption
and attention-merge CPU/CUDA columns. Other ranks do not write timing files;
there is no timing JSON or Chrome trace for this command.

After edge fine-tuning, use the exported edge checkpoint as follows.

The existing runner can use its built-in seven-camera reference recording and
derive Canny controls from the RGB videos:

```bash
python -m cosmos_framework.scripts.run_transfer_auto \
  --checkpoint "$EDGE_HF" \
  --control-type edge --frames 61 --resolution 480 \
  --mode camera --seed 0 \
  --output outputs/edge_61_camera
```

`camera` conditions on the initial RGB frame in each view. Use
`--mode text2video` for edge/caption-only generation: the runner uses source RGB
to compute edges, then removes RGB conditioning from the generated request.
Both paths retain the seven learned physical camera IDs.

With `--control-type edge`, the default frame count is 61. Add `--prepare-only`
to write and inspect controls/request/captions without downloading or loading
model weights. To use your own recordings, pass `--request /path/to/request.json`
with per-camera `vision_path` and `caption_path`. For already prepared edge
videos, use `--edge-source provided` and provide each camera's `control_path`.
The runner writes `edge: {}` and removes the WSM hint. The released WSM checkpoint
has not been edge-adapted; actual generation requires your exported fine-tune.

Step profiling remains available with `--profile-steps`; the default is one
generation with zero warmups. Cross-view attention remains enabled unless you
explicitly request the existing ablation flag.

CPU tests cover edge extraction, synchronized seven-view 61-frame formatting,
bundle requirements, fresh adaptation state and caption-task export. They do
not establish GPU training convergence or output quality.
