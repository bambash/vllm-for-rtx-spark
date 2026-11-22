"""Apply the Inkling 2 glue edits (vLLM main -> 0.32.0+spark) to an installed vllm package."""
import sys, pathlib
root = pathlib.Path(sys.argv[1])  # .../site-packages/vllm

def edit(rel, old, new, count=1):
    p = root / rel; s = p.read_text(encoding="utf-8")
    if new in s: print(f"skip  {rel}"); return
    n = s.count(old)
    if n != count: raise SystemExit(f"FAIL  {rel}: anchor found {n}x, wanted {count}\n{old!r}")
    p.write_text(s.replace(old, new), encoding="utf-8", newline="\n"); print(f"ok    {rel}")

ARCHS = '"Inkling2ForConditionalGeneration",\n                "Inkling2LoopForConditionalGeneration",\n'
edit("config/model.py",
     'elif arch in ("InklingForCausalLM", "InklingForConditionalGeneration"):',
     'elif arch in (\n                "InklingForCausalLM",\n                "InklingForConditionalGeneration",\n                ' + ARCHS + '            ):')
edit("config/model.py", 'overrides = [\n                "auto_gptq",', 'overrides = [\n                "inkling2",\n                "auto_gptq",')
edit("config/vllm.py", '        "InklingForConditionalGeneration",\n        "Kimi',
     '        "InklingForConditionalGeneration",\n        "Inkling2ForConditionalGeneration",\n        "Inkling2LoopForConditionalGeneration",\n        "Kimi')
edit("model_executor/layers/quantization/__init__.py", '    "deepseek_v4_fp8",\n', '    "deepseek_v4_fp8",\n    "inkling2",\n')
edit("model_executor/layers/quantization/__init__.py", '    from .auto_awq import AutoAWQConfig',
     '    from vllm.models.inkling2.quant_config import Inkling2QuantConfig\n\n    from .auto_awq import AutoAWQConfig')
edit("model_executor/layers/quantization/__init__.py", '        "deepseek_v4_fp8": deepseek_config,\n',
     '        "deepseek_v4_fp8": deepseek_config,\n        "inkling2": Inkling2QuantConfig,\n')
edit("model_executor/model_loader/weight_utils.py",
     '            and "quantized_layers" not in hf_quant_config\n        ):',
     '            and "quantized_layers" not in hf_quant_config\n        ) or (\n            # Inkling2\'s per-tensor precisions are only in hf_quant_config.json.\n            model_config.quantization == "inkling2" and "model" not in hf_quant_config\n        ):')
edit("model_executor/models/registry.py", '_MULTIMODAL_MODELS = {\n',
     '_MULTIMODAL_MODELS = {\n    "Inkling2LoopForConditionalGeneration": (\n        "vllm.models.inkling2",\n        "Inkling2LoopForConditionalGeneration",\n    ),\n    "Inkling2ForConditionalGeneration": (\n        "vllm.models.inkling2",\n        "Inkling2ForConditionalGeneration",\n    ),\n')
edit("transformers_utils/config.py", '    inkling_model="InklingModelConfig",\n',
     '    inkling_model="InklingModelConfig",\n    inkling2="Inkling2MMConfig",\n    inkling2_loop="Inkling2LoopMMConfig",\n    inkling2_model="Inkling2ModelConfig",\n')
edit("transformers_utils/configs/__init__.py", '    "InklingMMConfig": "vllm.models.inkling.configs",\n',
     '    "InklingMMConfig": "vllm.models.inkling.configs",\n    "Inkling2ModelConfig": "vllm.models.inkling2.configs",\n    "Inkling2MMConfig": "vllm.models.inkling2.configs",\n    "Inkling2LoopMMConfig": "vllm.models.inkling2.configs",\n')
edit("transformers_utils/configs/__init__.py", '    "InklingMMConfig",\n]',
     '    "InklingMMConfig",\n    "Inkling2ModelConfig",\n    "Inkling2MMConfig",\n    "Inkling2LoopMMConfig",\n]')
edit("model_executor/layers/fused_moe/routed_experts_capturer.py",
     '    num_layers = model_config.get_total_num_hidden_layers()\n',
     '    # One row per layer visit for models that run some decoder layers several\n    # times per token (Inkling2 loop checkpoints).\n    num_layers = (\n        getattr(model_config.hf_text_config, "num_layer_visits", None)\n        or model_config.get_total_num_hidden_layers()\n    )\n')

# Chat params: Inkling 2's renderer reads `parallel_tool_calls` (#67325).
edit("renderers/params.py",
     '    tool_choice: Any | None = None\n    """Request-level tool choice for renderers that need API metadata."""\n',
     '    tool_choice: Any | None = None\n    """Request-level tool choice for renderers that need API metadata."""\n\n    parallel_tool_calls: bool = True\n    """Request-level `parallel_tool_calls` for renderers that need API metadata."""\n')
edit("renderers/params.py", '            tool_choice=self.tool_choice,\n',
     '            tool_choice=self.tool_choice,\n            parallel_tool_calls=self.parallel_tool_calls,\n')
for f in ("entrypoints/openai/chat_completion/protocol.py", "entrypoints/openai/responses/protocol.py"):
    edit(f, '            tool_choice=self.tool_choice if self.tools else None,\n',
         '            tool_choice=self.tool_choice if self.tools else None,\n            parallel_tool_calls=self.parallel_tool_calls is not False,\n')

# Inkling 1 conv state, extended upstream for Inkling 2 (#67325, #51951).
CS = "models/inkling/nvidia/sconv_swa_attn.py"
edit(CS, "        kernel_size: int,\n        prefix: str,\n    ) -> None:",
         "        kernel_size: int,\n        prefix: str,\n        replicate_hidden: bool = False,\n    ) -> None:")
edit(CS, "        hidden_per_head = hidden_size // num_kv_heads\n",
         "        # With ``replicate_hidden`` each rank spreads the full hidden width\n"
         "        # over its local heads (Inkling 2), so D grows with TP.\n"
         "        hidden_per_head = hidden_size // (\n"
         "            self.num_kv_heads if replicate_hidden else num_kv_heads\n"
         "        )\n")
edit(CS, "    def forward(self): ...\n",
         "    def forward(self): ...\n\n"
         "    @property\n"
         "    def cache_block_size(self) -> int:\n"
         "        \"\"\"Return the block size used by cache metadata and physical indexing.\"\"\"\n"
         "        if self.kv_cache.numel() > 0:\n"
         "            return self.kv_cache.shape[2]\n"
         "        return self.block_size\n")

# Post-load hooks: upstream runs every module's no-arg `process_weights_after_loading`
# children-first (`finalize_model_weights`); 0.32 only runs attention layers (with a
# dtype) and the top model. Inkling 2 (MoE repack, loop controller) needs the former,
# so models that set `finalize_children_first = True` take the upstream path.
edit("model_executor/model_loader/utils.py",
     "    # Initialize post-load attention weights for any attention layer and MM\n",
     "    if getattr(model, \"finalize_children_first\", False):\n"
     "        from vllm.model_executor.model_loader.weight_utils import (\n"
     "            finalize_model_weights,\n"
     "        )\n\n"
     "        finalize_model_weights(model, target_device)\n"
     "        return\n\n"
     "    # Initialize post-load attention weights for any attention layer and MM\n")
FMW = '''

def finalize_model_weights(
    model: torch.nn.Module, target_device: torch.device | None = None
) -> None:
    """Run every module's own post-load hook after a full load or reload, each
    after all of its submodules (backported from vLLM main for Inkling 2).

    Quantization methods and the submodules of quantized layers are skipped:
    their quant method processes them, in the loader's per-layer pass.
    """
    from collections.abc import Iterable

    from vllm.model_executor.layers.attention import is_deferred_attention_layer
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )
    from vllm.model_executor.model_loader.utils import device_loading_context

    seen: set[torch.nn.Module] = set()

    def children_first(module: torch.nn.Module) -> Iterable[torch.nn.Module]:
        quantized = isinstance(
            getattr(module, "quant_method", None), QuantizeMethodBase
        )
        for child in [] if quantized else module.children():
            if child not in seen:
                seen.add(child)
                yield from children_first(child)
        yield module

    for module in children_first(model):
        if isinstance(module, QuantizeMethodBase):
            continue
        hook = getattr(module, "process_weights_after_loading", None)
        if callable(hook):
            # 0.32 device_loading_context needs a device; only attention hooks
            # need offloaded weights moved onto it.
            if target_device is not None and is_deferred_attention_layer(module):
                with device_loading_context(module, target_device):
                    hook()
            else:
                hook()
'''
p = root / "model_executor/model_loader/weight_utils.py"
s = p.read_text(encoding="utf-8")
if "def finalize_model_weights(" not in s:
    p.write_text(s.rstrip("\n") + "\n" + FMW, encoding="utf-8", newline="\n"); print("ok    weight_utils finalize_model_weights")
else:
    print("skip  weight_utils finalize_model_weights")

# ---- Edits to the files copied from upstream main ----
A = "models/inkling2/nvidia/attention.py"
edit(A, """from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheLayerConfig""",
"""# vLLM 0.32 backport: allocation policy still lives on the spec classes.
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)""")
edit(A, """) -> KVCacheLayerConfig:
    spec = AttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        ),
    )
    if sliding_window is not None:
        return SlidingWindowManager.layer_config(spec, sliding_window=sliding_window)
    return FullAttentionManager.layer_config(spec)""",
""") -> KVCacheSpec:
    kwargs = dict(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        ),
    )
    if sliding_window is not None:
        return SlidingWindowSpec(**kwargs, sliding_window=sliding_window)
    return FullAttentionSpec(**kwargs)""")
edit(A, "def get_kv_cache_config(self, vllm_config: VllmConfig) -> KVCacheLayerConfig:",
        "def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:", count=2)
edit("models/inkling2/nvidia/model.py",
     "    # LoRA (``lora.py``): adapters use the checkpoint's module names.\n    packed_modules_mapping = PACKED_MODULES_MAPPING\n",
     "    # vLLM 0.32 backport: run every module's post-load hook children-first,\n"
     "    # as upstream's `finalize_model_weights` does.\n"
     "    finalize_children_first = True\n\n"
     "    # LoRA (``lora.py``): adapters use the checkpoint's module names.\n    packed_modules_mapping = PACKED_MODULES_MAPPING\n")
edit("parser/inkling.py", "from vllm.v1.token_buffer import rfind_tokens\n", '''

def rfind_tokens(token_ids, targets, start=0, stop=None):
    """vLLM 0.32 backport of `vllm.v1.token_buffer.rfind_tokens`: index of the
    last token in `token_ids[start:stop]` that is one of `targets`, or -1."""
    if stop is None:
        stop = len(token_ids)
    wanted = {t for t in targets if t is not None}
    for i in range(stop - 1, start - 1, -1):
        if token_ids[i] in wanted:
            return i
    return -1

''')

# FlashInfer helper used by inkling2 ops/linear.py.
p = root / "utils/flashinfer.py"; s = p.read_text(encoding="utf-8")
if "def is_flashinfer_autotuning(" not in s:
    s = s.rstrip("\n") + '''


# Inkling 2 backport (vLLM main): used by vllm.models.inkling2 to skip work
# while a FlashInfer autotune pass is running.
def is_flashinfer_autotuning() -> bool:
    """Whether a FlashInfer autotune(True) context is active."""
    try:
        from flashinfer.autotuner import AutoTuner
    except Exception:
        return False
    return AutoTuner.get().is_tuning_mode
'''
    p.write_text(s, encoding="utf-8", newline="\n"); print("ok    utils/flashinfer.py")
else:
    print("skip  utils/flashinfer.py")
