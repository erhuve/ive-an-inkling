# The Inkomplete Inkling Persona Atlas

A text-only recreation of Joshua Fonseca Rivera’s [Neuron Persona Library](https://neuronpersonas.com/) for [Thinking Machines Lab’s Inkling-Small](https://thinkingmachines.ai/model-card/inkling-small/).

## Open the viewer

This repository contains the experiment runners and a standalone viewer in `atlas/`. Serve it from the repository root.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory atlas
```

Open [the local atlas](http://127.0.0.1:8765) to browse the bundled BF16 pilot in `atlas/run.json`. Select **Import run** to explore another runner’s JSON output in the browser.

The published cards have **Copy link** buttons that point to a dedicated page for each reply and direction. Each page includes the selected reply and its own social preview. Imported runs stay in the browser. To generate the static site with its reply pages, install Pillow in a virtual environment and build into an empty directory:

```sh
python -m pip install Pillow
atlas_build=$(mktemp -d)
python -m experiment.build_atlas "$atlas_build"
```

Serve or upload the contents of that directory as the site root.

The bundled atlas covers 133 neurons and 808 attempts. The first five dense neurons yielded 40 incoherent attempts at 0.5× through 1.25×. The later sweep tested 48 new coordinates across layers 0, 1, 8, 16, 24, 32, 35, and 41, with six coordinates per layer and both directions at 0.5×, 0.25×, and 0.1×. Coordinates were sampled with seed `20261105`. The sweep included 12 dense, 12 shared-expert, and 24 routed-expert coordinates and used batches of 12.

The [targeted shard](results/sweep/sweep-00060137-03da2a5563f138a4.partial.json) added both directions for 80 neurons in layer 16, routed expert 128 (neurons 1152–1231). It used the same three strengths and batch size as the 48-neuron sweep. Modal reached the 20-minute limit, so this file is the saved progress for shard 60137 of the 128-coordinate partition. It is a targeted sample near the pilot's Inky result.

The library shows personas, changed assistants, and coherent off-topic replies first. One example of each named finding appears ahead of its variants and repeated text. The overview features the highest ranked reply that is not marked sensitive. Open a card and expand **All attempts** to compare strengths. The [original run](results/pilot-1.25.json), [lower-strength run](results/pilot-lower-strengths.json), [four-neuron run](results/dense-four.json), and [48-neuron sweep](results/stratified-48.json) contain the raw replies, token IDs, and settings.

The library supports full-text search, filters, neuron lookup, push direction comparison and CSV export. The atlas shows all 42 layers and the observations loaded for each direction. Use a single pilot or a bounded subset in the viewer. A complete sweep would require a different storage and indexing strategy.

## Experiment

Ask “Who are you?” using the official text chat template with reasoning disabled. On the unsteered prompt, measure the median L2 norm of the residual stream after the selected block, excluding the first prompt token. At each prompt and generation position, add or subtract a normalized MLP down-projection column scaled by that norm and the chosen strength. Decode greedily and store the full token IDs, original text, assistant reply, and stop reason.

Dense layers 0 and 1 have 16,384 neuron coordinates each. In layers 2–41, every shared or routed expert has 2,048 coordinates. A full sweep would cover 21,168,128 coordinates and 42,336,256 pushes in both directions. The direct post-block intervention uses the expert’s direction regardless of routing and bypasses the selected block’s MLP convolution. [Reference protocol](https://neuronpersonas.com/#method) · [Inkling-Small architecture](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/config.json)

The default pilot tests four reproducibly sampled neurons in each dense layer, in both directions at strength 1.25. Pass `--unit` and `--strength` repeatedly to choose additional coordinates or strengths. The raw runner retains all requested strengths and puts the strongest first. In the reviewed atlas, each direction can surface an interesting weaker attempt. Generation stops at EOS or the token cap. Category, persona name, description, tags, weirdness, coherence and sensitive labels can be supplied separately in the JSON.

## Run on Apple Silicon

The pinned [mixed-precision MLX checkpoint](https://huggingface.co/Sawfwair/Inkling-Small-MLX-Mixed-2bit) uses BF16 for dense and shared projections and 2-bit weights for routed experts. This path targets a 128 GB Apple Silicon Mac and downloads about 84.5 GB. The viewer identifies its results as a distinct checkpoint.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r experiment/requirements-mlx.txt
python experiment/run_mlx.py --unit 0:dense:12623 --max-new-tokens 64 --output inkling-small-mlx-pilot.json
```

## Run on a CUDA host

The official BF16 checkpoint needs roughly 532 GB of storage and at least 600 GB of aggregate GPU memory. Install a CUDA-enabled PyTorch build for the host, then run a one-neuron pilot.

```sh
python -m pip install 'transformers==5.22.1' accelerate scipy
python experiment/run.py --unit 0:dense:12623 --max-new-tokens 64 --output inkling-small-bf16-pilot.json
```

Use `--plan` on either runner to preview the selected coordinates and settings without loading weights. Use `--resume` with the same settings and output path to continue a partial local run.

## Run a BF16 pilot on Modal

Modal requires a payment method on file for GPU jobs. If you intend to use only included Starter compute, confirm your remaining credit, leave a buffer for startup and resource charges, and set the workspace’s out-of-pocket spend limit to $0 in Usage & Billing. [GPU requirements](https://modal.com/docs/guide/gpu) · [Pricing](https://modal.com/pricing) · [Spend limits](https://modal.com/docs/guide/budgets)

```sh
python -m pip install 'modal==1.6.3'
modal setup
modal run -m experiment.modal_run --unit 0:dense:12623 --max-new-tokens 64 --output inkling-small-modal.json
```

The entrypoint stages the pinned BF16 weights in a reusable Modal Volume on CPU, with a 60-minute execution limit for staging. It then allocates four B300s for up to 20 minutes and returns the unsteered reply and both push directions. At the listed B300 rate, 20 minutes costs about $9.47 in GPU usage before CPU and memory. Use `--prepare-only` to stage the checkpoint separately. The first run needs enough credit for staging and inference. [Weight staging](https://modal.com/docs/guide/model-weights) · [Pricing](https://modal.com/pricing)

Once the checkpoint is staged, use `--gpu-only` and `--strengths` to run the three lower strengths for the same coordinate. The follow-up saves a separate JSON file.

```sh
modal run -m experiment.modal_run --gpu-only --unit 0:dense:12623 \
  --strengths 1.0,0.75,0.5 --max-new-tokens 64 --output inkling-small-modal-weaker.json
```

Pass comma-separated coordinates to `--units` to test several neurons in one GPU job. Set `--run-id` to a stable identifier if you want to resume the same plan. The runner saves completed directions to a partial file on its results Volume and prints the recovery command.

Use `--batch-size` from 1 to 16 to generate several interventions at the same layer together. The default is 1, and the unsteered baseline always runs alone. The runner records the batch size for batched runs because generation can vary with batch composition. [Reference limitations](https://neuronpersonas.com/#method)

## Plan the full sweep

The planner indexes every neuron coordinate in layer, expert, and neuron order. Pick one shard size for an exhaustive pass. A plan prints the selected coordinates and a stable run ID tied to the exact generation settings. Planning does not use Modal credits.

```sh
python3 -m experiment.sweep summary
python3 -m experiment.sweep plan --units-per-shard 128 --shard 0
python3 -m experiment.sweep plan --units-per-shard 128 --shard 0 --format modal
python3 -m experiment.sweep_status results
```

Run the printed Modal command after checking the workspace budget and staging the checkpoint. Repeat the same command to resume a shard that did not finish. The planner can also print the recovery commands for the final file (`--format recover`) or saved progress (`--format partial`). Keep recovered files under `results/` to include them in the coverage count. The status command deduplicates directions across shards and older pilots; it counts a direction as observed once it has a reply at any strength.

The default three strengths require 127,008,768 generations. On the 48-neuron pilot, the warm portion of a four-B300 job processed about 0.33 directions per second. At that rate and the listed B300 price, an exhaustive pass would take about four years on one worker and cost roughly $1 million in GPU time alone, before startup, storage, and review. The standalone atlas is for browsing a curated subset of results.

Before funding a full pass, benchmark [grouped expert dispatch](https://huggingface.co/docs/transformers/en/experts_interface) and the [native causal convolution kernel](https://github.com/Dao-AILab/causal-conv1d/tree/v1.7.0) independently. Use the same BF16 checkpoint, coordinates, strengths, and batch layout, then compare token IDs, reviewed categories, and warm throughput.
