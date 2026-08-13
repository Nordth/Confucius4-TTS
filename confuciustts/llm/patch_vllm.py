"""Runtime patches that make vLLM serve the Confucius4-TTS T2S model.

Importing this module has two side effects, both required before an engine is
created:

1. Registers the custom ``Text2SemanticVLLM`` architecture in vLLM's model
   registry so it can be loaded by name.
2. Monkeypatches ``GPUModelRunner._prepare_inputs`` so that, for this model,
   position ids are shifted to be relative to the start of the *semantic*
   sequence (excluding the speaker/text/BOS prefix). The T2S positional
   embeddings are trained on that convention, so without this correction the
   generated audio would be wrong.
"""

from vllm import ModelRegistry
from confuciustts.llm.llm_vllm import Text2SemanticVLLM

ModelRegistry.register_model("Text2SemanticVLLM", Text2SemanticVLLM)
print("Registered Text2SemanticVLLM into vLLM ModelRegistry")


def register_models():
    # Kept as an explicit no-op entry point: importing this module already does
    # the registration, but callers can reference this to force the import.
    pass

import numpy as np
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

# vLLM's own _prepare_inputs. We wrap it (instead of shipping a copy of the
# implementation) because its internals change between vLLM releases.
_ORIGINAL_PREPARE_INPUTS = GPUModelRunner._prepare_inputs


def _prepare_inputs(
    self,
    scheduler_output: "SchedulerOutput",
    num_scheduled_tokens: np.ndarray,
) -> tuple[torch.Tensor, SpecDecodeMetadata | None]:
    """vLLM's GPUModelRunner._prepare_inputs + Text2SemanticVLLM position shift.

    Delegates to the installed vLLM's implementation, then shifts the scheduled
    positions of Text2SemanticVLLM prefill requests so the semantic sequence is
    indexed from 0 (the last prompt position carries the BOS prefix embedding).
    The KV slot mapping is left untouched (it is computed from the unshifted
    positions inside the original implementation); only the position ids passed
    to the model forward are corrected.
    """
    logits_indices, spec_decode_metadata = _ORIGINAL_PREPARE_INPUTS(
        self, scheduler_output, num_scheduled_tokens
    )

    model = self.get_model()
    if not isinstance(model, Text2SemanticVLLM):
        return logits_indices, spec_decode_metadata

    # Only the prefill of a request that carries audio prefix embeddings needs
    # the shift: the whole scheduled prompt is then the (speaker + text + BOS)
    # prefix, whose semantic positions start at 0. Decode steps of the same
    # request (and any non-multimodal request) are not touched.
    per_req_offsets = np.zeros(self.input_batch.num_reqs, dtype=np.int64)
    for new_req in scheduler_output.scheduled_new_reqs:
        if not new_req.mm_features or not new_req.prompt_token_ids:
            continue
        req_index = self.input_batch.req_id_to_index.get(new_req.req_id)
        if req_index is None or req_index >= self.input_batch.num_reqs:
            continue
        per_req_offsets[req_index] = -(len(new_req.prompt_token_ids) - 1)

    offsets_np = np.repeat(per_req_offsets, num_scheduled_tokens)
    if np.any(offsets_np):
        offsets = torch.from_numpy(offsets_np).to(self.device, non_blocking=True)
        self.positions[: scheduler_output.total_num_scheduled_tokens] += offsets

    return logits_indices, spec_decode_metadata


# Install the patched method in place of vLLM's original.
GPUModelRunner._prepare_inputs = _prepare_inputs
print("GPUModelRunner._prepare_inputs patched for Confucius4-TTS position correction")
