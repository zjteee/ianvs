# Cloud-Edge Speculative Decoding Benchmark

This example implements a cloud-edge speculative decoding benchmark on top of the Ianvs `jointinference` paradigm. It is used to evaluate token-level collaboration between an edge-side `drafter` and a cloud-side `verifier`, and to compare that collaborative path against `edge-only` and `cloud-only` baselines under the same benchmark pipeline.

## What This Example Provides

- A complete speculative decoding benchmark example based on Ianvs and Sedna `JointInference`.
- Dedicated `drafter` and `verifier` modules for token-level collaboration.
- Two algorithm lines under the same Ianvs `jointinference` interface:
  - `AR`: token-level speculative decoding
  - `block`: DFlash-style block speculative decoding
- Three comparable execution modes:
  - `collaboration`
  - `cloud-only`
  - `edge-only`
- Built-in benchmark metrics for latency, throughput, and acceptance rate.
- Configurable model pairing, draft window size, dataset, and runtime profile.

## Key Files

- `examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob.yaml`
  - Benchmark entrypoint and ranking configuration.
- `examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml`
  - Dataset and metric definitions.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding.yaml`
  - Algorithm modules and main hyperparameter matrix.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/data_processor.py`
  - Converts dataset samples into the request format used by the benchmark.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/ar/drafter.py`
  - Edge-side draft generation logic.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/ar/verifier.py`
  - Cloud-side verification and correction logic.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/block/drafter.py`
  - Edge-side DFlash draft wrapper.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/block/verifier.py`
  - Cloud-side DFlash verification wrapper.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/profiles/base.yaml`
  - Shared runtime profile, including network-delay simulation and generation settings.
- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding_block.yaml`
  - Block/DFlash algorithm configuration.
- `examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob_block.yaml`
  - Benchmark entrypoint for the block/DFlash path.

## Execution Modes

The benchmark uses the `drafter` module hyperparameter `inference_mode` to switch among three execution paths:

- `collaboration`
  - Edge drafts several tokens and cloud verifies them round by round.
- `cloud-only`
  - Uses the verifier-side model as the standalone baseline.
- `edge-only`
  - Uses the drafter-side model as the standalone baseline.

This keeps all three modes under the same benchmark interface, which makes the results directly comparable.

For the `block` path, only `collaboration` and `cloud-only` are supported. `edge-only` is not defined because the DFlash draft model depends on target hidden states.

## Dataset Configuration

The example now vendors the benchmark datasets it needs under its own directory, so reproduction does not depend on a separate repository-level dataset checkout. The included datasets are:

- `examples/cloud-edge-speculative-decoding-benchmark/dataset/gsm8k`
- `examples/cloud-edge-speculative-decoding-benchmark/dataset/humaneval`

The default test dataset entry is configured in `examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml`. The sample transformation logic is implemented in `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/data_processor.py`.

The dataset processor also supports:

- `sample_size`
  - Limits how many test samples are used in one run.

## Default Model Setup

The default `AR` benchmark configuration uses:

- `drafter`: `Qwen/Qwen2.5-0.5B-Instruct`
- `verifier`: `Qwen/Qwen2.5-7B-Instruct`

These defaults are defined in `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding.yaml`.

The default `block` benchmark configuration uses:

- `drafter`: `z-lab/Qwen3-8B-DFlash-b16`
- `verifier`: `Qwen/Qwen3-8B`

These defaults are defined in `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding_block.yaml`.

## Main Configurable Variables

The benchmark is designed so that baseline and collaborative runs can be driven by the same configuration structure. The most important variables are:

- `inference_mode`
  - Chooses `collaboration`, `cloud-only`, or `edge-only`.
- `generation_backend`
  - For standalone baseline generation, chooses `custom` or `transformers`.
  - `custom` is the hand-written token-by-token decoding loop used by the speculative path.
  - `transformers` uses `transformers.generate`.
- `block_cloud_only_backend`
  - For `block cloud-only`, chooses `custom` or `transformers`.
- `draft_tokens_per_step`
  - Controls how many speculative draft tokens are proposed in each collaboration round.
- `model`
  - The drafter and verifier models are independently configurable.
- `sample_size`
  - Controls the number of evaluated samples.
- Runtime profile settings in `profiles/base.yaml`
  - Prompt and completion limits
  - Temperature and generation controls
  - Device and backend options
  - Network delay simulation such as `enable_network_sleep`, `network_rtt_ms`, and `network_jitter_ms`

For collaborative speculative decoding, the example uses the custom paper-aligned implementation. Backend switching is only intended for standalone baseline generation where the wrapper explicitly supports it.

For the `block` path, the example also expects:

- `attn_implementation`
  - Attention backend passed to both the block draft model and target model.

The required DFlash draft-model implementation is vendored directly under:

- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/block/copied_dflash_model.py`
  - Local copied-and-trimmed implementation of the upstream draft model used by the block path.

## Metrics

The benchmark reports:

- `Time to First Token`
- `Throughput`
- `Internal Token Latency`
- `End-to-End Latency`
- `Acceptance Rate`

Metric definitions are configured in `examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml`.

## Run the Benchmark

Run from the Ianvs repository root:

```bash
ianvs -f examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob.yaml
```

Run the block/DFlash path with:

```bash
ianvs -f examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob_block.yaml
```

If models need to be downloaded through the Hugging Face mirror in mainland China, run:

```bash
HF_ENDPOINT=https://hf-mirror.com \
HUGGINGFACE_HUB_BASE_URL=https://hf-mirror.com \
ianvs -f examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob.yaml
```

## Output

Benchmark artifacts are written into the workspace configured in `benchmarkingjob.yaml`. Typical outputs include:

- benchmark ranking tables
- per-run metric results
- `specdec_sample_outputs.jsonl` with prompt / output / reference records for each sample

## Notes

- This example focuses on the speculative decoding workflow and benchmark evaluation, rather than distributed deployment packaging.
- The current execution path is an in-process cloud-edge collaboration simulation, which is useful for validating workflow correctness and benchmarking collaboration cost.
- The benchmark-specific logic is isolated in this example directory and does not require redefining the semantics of the existing generic `cloud` and `edge` modules used by other Ianvs examples.
