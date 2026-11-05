# Inkling Persona Atlas

A text-only recreation of Joshua Fonseca Rivera’s [Neuron Persona Library](https://neuronpersonas.com/) for [Thinking Machines Lab’s Inkling-Small](https://thinkingmachines.ai/model-card/inkling-small/).

## Open the viewer

This repository contains the experiment runners and a standalone viewer in `atlas/`. Serve it from the repository root.

```sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory atlas
```

Open [the local atlas](http://127.0.0.1:8765) to browse the bundled BF16 pilot in `atlas/run.json`. Select **Import run** to explore another runner’s JSON output in the browser. Use `atlas/` as the document root on a static host.

The bundled pilot covers dense layer 0, neuron 12,623, in both directions at 1.25×, 1.0×, 0.75×, and 0.5×. All eight attempts were manually reviewed as incoherent, so the cards display the original 1.25× replies. Open either card and expand **All attempts** to compare the strengths. The [original run](results/pilot-1.25.json) and the [lower-strength run](results/pilot-lower-strengths.json) contain the raw replies, token IDs, and settings.

The library supports full-text search, filters, neuron lookup, push direction comparison and CSV export. The atlas shows all 42 layers and the observations loaded for each direction. Use a single pilot or a bounded subset in the viewer. A complete sweep would require a different storage and indexing strategy.

## Experiment

Ask “Who are you?” using the official text chat template with reasoning disabled. On the unsteered prompt, measure the median L2 norm of the residual stream after the selected block, excluding the first prompt token. At each prompt and generation position, add or subtract a normalized MLP down-projection column scaled by that norm and the chosen strength. Decode greedily and store the full token IDs, original text, assistant reply, and stop reason.

Dense layers 0 and 1 have 16,384 neuron coordinates each. In layers 2–41, every shared or routed expert has 2,048 coordinates. A full sweep would cover 21,168,128 coordinates and 42,336,256 pushes in both directions. The direct post-block intervention uses the expert’s direction regardless of routing and bypasses the selected block’s MLP convolution. [Reference protocol](https://neuronpersonas.com/#method) · [Inkling-Small architecture](https://huggingface.co/thinkingmachines/Inkling-Small/blob/main/config.json)

The default pilot tests four reproducibly sampled neurons in each dense layer, in both directions at strength 1.25. Pass `--unit` and `--strength` repeatedly to choose additional coordinates or strengths. The runner retains all requested strengths and shows the strongest in the library. Generation stops at EOS or the token cap. Category, persona name, description, tags, weirdness, coherence and sensitive labels can be supplied separately in the JSON.

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

Modal requires a payment method on file for GPU jobs. If you intend to use only included Starter compute, confirm your remaining credit and set the workspace’s out-of-pocket spend limit to $0 in Usage & Billing. [GPU requirements](https://modal.com/docs/guide/gpu) · [Pricing](https://modal.com/pricing) · [Spend limits](https://modal.com/docs/guide/budgets)

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
