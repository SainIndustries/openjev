"""Clef and Clef-flash as a vLLM pooling model: the backbone in vLLM, Cloudflare's joint schema
head in the pooler.

From kurcontko's vLLM plugin (huggingface.co/kurcontko/clef-NVFP4, vllm_plugin/clef_vllm/model.py,
Apache-2.0), with the vision tower kept and the head found beside the weights or in clef_head/.
The head is the release's own JointSchemaHead, from the checkpoint's joint_schema_model.py.

The pooler gathers a request's final hidden states across chunked prefill and runs the head
on the whole prompt. Each request carries its layout in PoolingParams.extra_kwargs
["clef_questions"] and its length as Clef counts it in ["clef_length"]
(openjev.clef_plugin.schema_layout). The output is one flat tensor: every option logit of
every question, in order, or an empty one when vLLM's prompt does not line up with Clef's.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable, Set
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from .clef_plugin import TASK


def release_module(model_dir: Path):
    """The checkpoint's joint_schema_model.py, as a module."""
    spec = importlib.util.spec_from_file_location("joint_schema_model", model_dir / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def head_dir(model_dir: Path) -> Path:
    """kurcontko's checkpoints keep the head in clef_head/, so vLLM does not read it as backbone weights."""
    return model_dir / "clef_head" if (model_dir / "clef_head").is_dir() else model_dir


class ClefPooler(Pooler):
    def __init__(self, model: "ClefForDecision") -> None:
        super().__init__()
        self._model = [model]  # not a submodule: the model owns and loads the head
        self.enable_chunked_prefill = get_current_vllm_config().scheduler_config.enable_chunked_prefill

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {TASK}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata):
        model = self._model[0]
        cursor = pooling_metadata.get_pooling_cursor()
        chunks = list(torch.split(hidden_states, cursor.num_scheduled_tokens_cpu.tolist()))
        if self.enable_chunked_prefill:
            finished = cursor.is_finished().tolist()
            full: list[torch.Tensor | None] = []
            for state, chunk, done in zip(pooling_metadata.pooling_states, chunks, finished):
                state.hidden_states_cache.append(chunk if done and not state.hidden_states_cache else chunk.clone())
                if done:
                    cache = state.hidden_states_cache
                    full.append(cache[0] if len(cache) == 1 else torch.cat(cache, dim=0))
                    state.clean()
                else:
                    full.append(None)
        else:
            full = chunks

        outputs: list[torch.Tensor | None] = []
        token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        for hidden, ids, params in zip(full, token_ids, pooling_metadata.pooling_params):
            if hidden is None:
                outputs.append(None)
                continue
            if model.lexical_weight is None:  # a tensor-parallel rank that does not pool
                outputs.append(torch.zeros(2, device=hidden.device))
                continue
            extra = params.extra_kwargs or {}
            if extra.get("clef_length", hidden.shape[0]) != hidden.shape[0]:
                # vLLM expanded an image to another number of tokens than Clef's processor did, so
                # the spans would point at the wrong tokens. Raising here would stop the engine.
                outputs.append(torch.zeros(0, device=hidden.device))
                continue
            # vLLM's profiling run has no schema: score a dummy one
            layout = extra.get("clef_questions") or [[1, [0, 1], [[0, 1], [0, 1]]]]
            outputs.append(model.score(hidden, ids, layout))
        return outputs


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class ClefForDecision(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    default_tok_pooling_type = "ALL"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.model_dir = Path(vllm_config.model_config.model)
        self.release = release_module(self.model_dir)
        head_config = json.loads((head_dir(self.model_dir) / "joint_head_config.json").read_text())
        self.head = self.release.JointSchemaHead(**head_config)
        self.lexical_weight = None
        self.pooler = ClefPooler(self)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        state = load_file(head_dir(self.model_dir) / "joint_head.safetensors")
        self.head.load_state_dict(state, strict=True)
        lm_head = self.language_model.lm_head.weight
        self.head.to(device=lm_head.device, dtype=lm_head.dtype).eval()
        # The head reads lm_head rows by token id. Under tensor parallelism the weight is split
        # by vocabulary, so rank 0 (the one that pools) gathers it once into host memory, and
        # a request copies only its option tokens' rows to the GPU (see score).
        self.lexical_weight = lm_head.data
        if get_tensor_model_parallel_world_size() > 1:
            gathered = tensor_model_parallel_all_gather(lm_head.data.contiguous(), dim=0)
            self.lexical_weight = gathered.cpu().pin_memory() if get_tensor_model_parallel_rank() == 0 else None
            del gathered
            torch.cuda.empty_cache()
        return loaded | {f"head.{name}" for name in state}

    @torch.inference_mode()
    def score(self, hidden: torch.Tensor, ids: torch.Tensor, layout: list) -> torch.Tensor:
        release = self.release
        questions = tuple(
            release.EncodedQuestion(
                question_id=str(index),
                question_type=int(type_id),
                question_span=tuple(question_span),
                option_spans=tuple(tuple(span) for span in option_spans),
                option_ids=tuple(str(i) for i in range(len(option_spans))),
            )
            for index, (type_id, question_span, option_spans) in enumerate(layout)
        )
        record = release.EncodedRecord(input_ids=tuple(), questions=questions, record_id="vllm")
        device = hidden.device
        lexical = self.lexical_weight
        if lexical.device.type == "cpu":
            # only the option spans' rows, renumbered into a compact table
            ids = ids.cpu()
            positions = torch.cat([torch.arange(start, end) for _, _, spans in layout for start, end in spans])
            unique, inverse = torch.unique(ids[positions], return_inverse=True)
            lexical = lexical[unique].to(device, non_blocking=True)
            ids = torch.zeros(ids.shape, dtype=torch.long)
            ids[positions] = inverse
        logits = self.head(
            hidden.unsqueeze(0),
            ids.to(device).unsqueeze(0),
            torch.ones((1, hidden.shape[0]), dtype=torch.long, device=device),
            [record],
            lexical,
        )[0]
        return torch.cat([question_logits.float() for question_logits in logits])
