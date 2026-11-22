# Inkling 2 / Inkling 2 Mini backport

Backports `thinkingmachines/Inkling-2-Mini` (`Inkling2LoopForConditionalGeneration`) and
Inkling 2 Small from vLLM main (`8c17e4692`, 2026-11-22) to this project's vLLM 0.32.0.
The result is `patches/vllm/0019-inkling2-mini.patch`; these files regenerate it and can
also patch an already-installed release in place.

| File | Purpose |
|---|---|
| `apply_inkling2.py` | `apply_inkling2.py <vllm package dir> <vllm main checkout>`: copies the upstream `vllm/models/inkling2` package, Inkling renderers and parser, then runs `apply_glue.py`. Re-runnable. |
| `apply_glue.py` | Every edit to 0.32 files and to the copied upstream files, as anchored string replacements that fail loudly if an anchor moves. |
| `inkling2-mini-test.yaml`, `serve-test.sh` | Test server on port 8001 (32K context, no DSpark). |

## What changed relative to upstream

- **KV cache API**: upstream's `get_kv_cache_config` / `XManager.layer_config` (#61452) mapped back to
  0.32's `get_kv_cache_spec` with `FullAttentionSpec` / `SlidingWindowSpec`.
- **Post-load hooks**: upstream runs every module's no-argument `process_weights_after_loading`
  children-first (`finalize_model_weights`). 0.32 only runs attention layers (with a dtype) and the
  top model, which would skip Inkling 2's MoE repack. Ported `finalize_model_weights`; models opt in
  with `finalize_children_first = True` (only Inkling 2 does), so other models are unchanged.
- **Inkling 1 conv state**: added upstream's `replicate_hidden` option and `cache_block_size`.
- **Registry and config glue**: architectures, `inkling2`/`inkling2_loop` configs, `inkling2` quant
  method, `hf_quant_config.json` loading, routed-experts layer visits, `parallel_tool_calls` in chat params.
- **Small helpers**: `is_flashinfer_autotuning`, and an inline `rfind_tokens`.

SM12x (RTX Spark) uses upstream's Triton paths, so no native kernels are needed: the native
`conv_mix`/all-reduce/seq-parallel CUDA ops are only used on SM90/SM100 or with tensor parallelism.

## Not ported

- **DSpark speculative decoding** (`vllm.models.inkling2.nvidia.dspark`, `speculator_config`): needs
  `vllm.v1.worker.gpu.spec_decode.dspark` and `sample.greedy_argmax` from newer vLLM.
- Rust frontend renderer changes (the Python renderer is used).

## Verified (2026-11-21, RTX Spark N1X, release 0.32.0+spark.1 with this patch applied in place)

- Loads in ~90 s (18.3 GiB weights); CUDA graphs capture.
- Chat ~40 tok/s short, ~37 tok/s sustained over 1,500 tokens; reasoning split into `reasoning`.
- Tool calling round trip (`tool-call-parser: inkling`), image input.
- Fork unit tests: the 10 failures under `-p '*_test.py'` are identical with and without this patch.

## Context length

The full 1,048,576-token context works: with `kv-cache-memory-bytes: 60000000000` vLLM reports
1,255,627 tokens of KV cache, and a 973K-token prompt recalled a fact placed 5% in (22 min prefill,
~740 tok/s; 154K tokens took 89 s). Reported capacity depends on `max-model-len` and `max-num-seqs`,
not only on memory (22.35 GiB gave 46,756 tokens at 32K), so measure rather than extrapolate.
