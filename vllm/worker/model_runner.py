import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils.communication_op import (
    broadcast, broadcast_object_list)
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.utils import in_wsl

logger = init_logger(__name__)

KVCache = Tuple[torch.Tensor, torch.Tensor]
_PAD_SLOT_ID = -1
# Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
# NOTE: _get_graph_batch_size needs to be updated if this list is changed.
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [8 * i for i in range(1, 33)]

def patch_model_with_openvino(model, model_config, *model_args, **model_kwargs):
    if hasattr(model, '_openvino_patch_orig_forward'):
        return
    # Replace forward with our stuff
    import openvino as ov

    import torch
    import numpy as np
    import openvino as ov
    from vllm.model_executor.layers.attention import PagedAttention
    from openvino.frontend.pytorch import ModuleExtension

    from typing import Optional

    import torch
    from dataclasses import dataclass

    @dataclass
    class InputMetadata:
        """Metadata for input sequences. Used in PagedAttention.

        Args:
            prompt_lens: Lengths of prompts.
            slot_mapping: The address to write the new KV to of each token.
            max_context_len: The maximum context length.
            context_lens: the length of attention context for each sequence.
            block_tables: The block tables. (Seq id -> list of physical block)
            kv_cache_dtype: Data type to store kv cache.
        """

        def __init__(
            self,
            is_prompt: bool = False,
            slot_mapping: torch.Tensor = None,
            prompt_lens: Optional[torch.Tensor] = None,
            max_seq_len: Optional[int] = None,
            start_loc: Optional[torch.Tensor] = None,
            max_context_len: Optional[int] = None,
            context_lens: Optional[torch.Tensor] = None,
            block_tables: Optional[torch.Tensor] = None,
            use_cuda_graph: bool = False,
            kv_cache_dtype: str = "auto",
        ) -> None:
            self.is_prompt = is_prompt
            self.prompt_lens = prompt_lens
            self.max_seq_len = max_seq_len
            self.start_loc = start_loc
            self.max_context_len = max_context_len
            self.slot_mapping = slot_mapping
            self.context_lens = context_lens
            self.block_tables = block_tables
            self.use_cuda_graph = use_cuda_graph
            self.kv_cache_dtype = kv_cache_dtype

            # Set during the execution of the first attention op.
            # FIXME(woosuk): This is a hack.
            self.attn_bias = None

        def __repr__(self) -> str:
            return ("InputMetadata("
                    f"is_prompt={self.is_prompt}, "
                    f"max_context_len={self.max_context_len}, "
                    f"slot_mapping={self.slot_mapping}, "
                    f"context_lens={self.context_lens}, "
                    f"block_tables={self.block_tables}, "
                    f"use_cuda_graph={self.use_cuda_graph}, "
                    f"kv_cache_dtype={self.kv_cache_dtype})")

    _PAD_SLOT_ID = -1

    pt_model = model
    _BATCH_SIZES_TO_CAPTURE = [2]
    max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
    input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.long)
    input_positions = torch.zeros(max_batch_size, 1,
                                        dtype=torch.long)
    slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long)
    slot_mapping.fill_(_PAD_SLOT_ID)
    context_lens = torch.ones(max_batch_size, dtype=torch.int32)

    max_context_len_to_capture = (
                model_config.max_context_len_to_capture
                if model_config is not None else 0)
    block_size = 8
    max_num_blocks = (max_context_len_to_capture + block_size -
                            1) // block_size
    graph_block_tables = np.zeros(
                (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    block_tables = torch.from_numpy(graph_block_tables)

    input_meta = {"is_prompt": torch.tensor(False), "slot_mapping": slot_mapping, "max_seq_len": torch.tensor(256), "max_context_len": torch.tensor(2048), "context_lens": context_lens, "block_tables": block_tables}

    fp_type = torch.float32

    #TODO: Take example tensors from model_args/model_kwargs
    kv_cache = [(torch.randn((3640, 12, 16, 16, 4), dtype=fp_type), torch.rand((3640, 12, 64, 16), dtype=fp_type))] * 12

    example_input = (torch.ones((1, 1), dtype=torch.long), torch.range(0, 10, dtype=torch.long).unsqueeze(0)[:, -1:], tuple(kv_cache), input_meta)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, position_ids, kv_cache, meta_dict):
            input_meta = InputMetadata(**meta_dict)
            return self.model(input_ids, position_ids, kv_cache, input_meta)


    model_wrapper = ModelWrapper(pt_model)

    num_heads = pt_model.config.num_attention_heads
    embed_dim = pt_model.config.hidden_size
    head_dim = embed_dim // num_heads

    ov_dtype_maping = {
        torch.bool: ov.Type.boolean,
        torch.float32: ov.Type.f32,
        torch.float16: ov.Type.f16,
        torch.bfloat16: ov.Type.bf16,
        torch.int32: ov.Type.i32,
        torch.int64: ov.Type.i64
    }


    def flattenize_inputs(inputs):
        """
        Helper function for making nested inputs flattens
        """
        flatten_inputs = []
        for input_data in inputs:
            if input_data is None:
                continue
            if isinstance(input_data, (list, tuple)):
                flatten_inputs.extend(flattenize_inputs(input_data))
            elif isinstance(input_data, dict):
                flatten_inputs.extend(flattenize_inputs(list(input_data.values())))
            else:
                flatten_inputs.append(input_data)
        return flatten_inputs

    flatten_input = flattenize_inputs(example_input)
    input_names = ["input_ids", "position_ids"]
    output_names = ["logits"]

    for i in range(12):
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])

    input_names.extend(list(input_meta))

    def wrapper(module, target_op, *args, **kwargs):
        return target_op(
                            args[0],
                            args[1],
                            args[2],
                            args[3],
                            args[4],
                            args[5].is_prompt,
                            args[5].slot_mapping,
                            args[5].max_context_len,
                            args[5].context_lens,
                            args[5].block_tables,
                            torch.tensor(module.scale)
                        )

    with torch.no_grad():
        print('>>>>>>>>>>>>> CONVERTING OV MODEL')
        ov_model =  ov.convert_model(
            model_wrapper,
            example_input=example_input,
            extension=[
                ModuleExtension(
                    PagedAttention,
                    extension=lambda module: 'PagedAttentionPlaceholder',
                    replacer=lambda module, *args, **kwargs: args[0],
                    wrapper=wrapper
                )
            ]
        )

        for input_name, input_data, input_tensor in zip(input_names, flatten_input, ov_model.inputs):
            if input_tensor.element_type.is_dynamic():
                input_tensor.get_node().set_element_type(ov_dtype_maping[input_data.dtype])
            if input_tensor.partial_shape.rank.is_dynamic:
                input_tensor.get_node().set_partial_shape(ov.PartialShape([-1]*input_data.ndim))
            input_tensor.get_tensor().set_names({input_name})

        for out_name, out in zip(output_names, ov_model.outputs):
            out.get_tensor().set_names({out_name})
        ov_model.validate_nodes_and_infer_types()
        ov.save_model(ov_model, "vllm_openvino_model.xml")
        print('>>>>>>>>>>>>> OV MODEL CONVERTED')
        print(ov_model)
    ov_compiled = ov.compile_model(ov_model)

    from functools import partial
    def wrapper(*args, **kwargs):
        print('MY WRAPPER')
        print(f'model class: {type(args[0])}')
        #for i, input in enumerate(args[1:]):
        #    print(f'[{i}]: {type(input)}')
        #for key, value in kwargs.items():
        #    print(f'{key}: {type(value)}')
        #result = args[0]._openvino_patch_orig_forward(*args[1:], **kwargs)
        input_metadata = kwargs['input_metadata']
        #print(dir(input_metadata))
        inputs = [
            kwargs['input_ids'],
            kwargs['positions'],
            *flattenize_inputs(kwargs['kv_caches']),
            input_metadata.is_prompt, input_metadata.slot_mapping, 256#, input_metadata.max_context_len, input_metadata.context_lens
        ]
        #for input in inputs:
        #    print(f'{input.dtype} wiht shape {input.shape}' if isinstance(input, torch.Tensor) else type(input))
        result = ov_compiled(inputs, share_outputs=False)
        #print(f'result: {type(result)}')
        return torch.from_numpy(result[0])
    print(' ============= PATCHING MODEL =============')
    model._openvino_patch_orig_forward = model.forward
    model.forward = partial(wrapper, model)
class ModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        is_driver_worker: bool = False,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.model = None
        self.block_size = None  # Set after initial profiling.
        self.device = self.model_config.device

        self.graph_runners: Dict[int, CUDAGraphRunner] = {}
        self.graph_memory_pool = None  # Set during graph capture.

        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)
        # When using CUDA graph, the input block tables must be padded to
        # max_context_len_to_capture. However, creating the block table in
        # Python can be expensive. To optimize this, we cache the block table
        # in numpy and only copy the actual input content at every iteration.
        # The shape of the cached block table will be
        # (max batch size to capture, max context len to capture / block size).
        self.graph_block_tables = None  # Set after initial profiling.
        # cache in_wsl result
        self.in_wsl = in_wsl()

    def load_model(self) -> None:
        self.model = get_model(self.model_config)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        max_num_blocks = (self.max_context_len_to_capture + block_size -
                          1) // block_size
        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), max_num_blocks), dtype=np.int32)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, List[int]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []

        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.append(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.append(list(range(prompt_len)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.append([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            slot_mapping.append([])
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)
            for i in range(prompt_len):
                if i < start_idx:
                    slot_mapping[-1].append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping[-1].append(slot)

        max_prompt_len = max(prompt_lens)
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_prompt_len,
                                             pad=0,
                                             dtype=torch.long,
                                             device=self.device)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_prompt_len,
                                                pad=0,
                                                dtype=torch.long,
                                                device=self.device)
        slot_mapping = _make_tensor_with_pad(slot_mapping,
                                             max_prompt_len,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=self.device)

        input_metadata = InputMetadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            max_context_len=None,
            context_lens=None,
            block_tables=None,
            use_cuda_graph=False,
        )
        return input_tokens, input_positions, input_metadata, prompt_lens

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[List[int]] = []
        input_positions: List[List[int]] = []
        slot_mapping: List[List[int]] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt

            seq_ids = list(seq_group_metadata.seq_data.keys())
            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append([generation_token])

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append([position])

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append([slot])

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        batch_size = len(input_tokens)
        max_context_len = max(context_lens)
        use_captured_graph = (
            not self.model_config.enforce_eager
            and batch_size <= _BATCH_SIZES_TO_CAPTURE[-1]
            and max_context_len <= self.max_context_len_to_capture)
        if use_captured_graph:
            # Pad the input tokens, positions, and slot mapping to match the
            # batch size of the captured graph.
            graph_batch_size = _get_graph_batch_size(batch_size)
            assert graph_batch_size >= batch_size
            for _ in range(graph_batch_size - batch_size):
                input_tokens.append([])
                input_positions.append([])
                slot_mapping.append([])
                context_lens.append(1)
                block_tables.append([])
            batch_size = graph_batch_size

        # When using CUDA graph, we don't need to make the tensors on the GPU
        # because they will be eventually copied to the designated GPU buffer.
        device = "cpu" if use_captured_graph or self.device.type == "cpu" else "cuda"
        pin_memory = use_captured_graph and not self.in_wsl and self.device.type == "cuda"
        input_tokens = _make_tensor_with_pad(input_tokens,
                                             max_len=1,
                                             pad=0,
                                             dtype=torch.long,
                                             device=device)
        input_positions = _make_tensor_with_pad(input_positions,
                                                max_len=1,
                                                pad=0,
                                                dtype=torch.long,
                                                device=device)
        slot_mapping = _make_tensor_with_pad(slot_mapping,
                                             max_len=1,
                                             pad=_PAD_SLOT_ID,
                                             dtype=torch.long,
                                             device=device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=device)

        if use_captured_graph:
            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.graph_block_tables[:batch_size]
            for i, block_table in enumerate(block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.tensor(input_block_tables, device="cuda")
        else:
            max_block_table_len = (max_context_len + self.block_size -
                                   1) // self.block_size
            block_tables = _make_tensor_with_pad(
                block_tables,
                max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=device,
            )

        input_metadata = InputMetadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            max_context_len=max_context_len,
            context_lens=context_lens,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )
        return input_tokens, input_positions, input_metadata

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        selected_token_start_idx = 0
        categorized_sample_indices = {t: [] for t in SamplingType}
        categorized_sample_indices_start_idx = 0

        max_prompt_len = max(prompt_lens) if prompt_lens else 1
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        categorized_sample_indices_start_idx)
                categorized_sample_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += max_prompt_len
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        range(categorized_sample_indices_start_idx,
                              categorized_sample_indices_start_idx + num_seqs))
                categorized_sample_indices_start_idx += num_seqs

        selected_token_indices = _async_h2d(selected_token_indices,
                                            dtype=torch.long,
                                            device=self.device,
                                            pin_memory=not self.in_wsl and not self.device.type == "cpu")
        categorized_sample_indices = {
            t: _async_h2d(seq_ids, dtype=torch.int, device=self.device,
                                            pin_memory=not self.in_wsl and not self.device.type == "cpu")
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata, SamplingMetadata]:
        if self.is_driver_worker:
            # NOTE: We assume that all sequences in the group are all prompts or
            # all decodes.
            is_prompt = seq_group_metadata_list[0].is_prompt
            # Prepare input tensors.
            if is_prompt:
                (input_tokens, input_positions, input_metadata,
                 prompt_lens) = self._prepare_prompt(seq_group_metadata_list)
            else:
                (input_tokens, input_positions, input_metadata
                 ) = self._prepare_decode(seq_group_metadata_list)
                prompt_lens = []
            sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens)

            def get_size_or_none(x: Optional[torch.Tensor]):
                return x.size() if x is not None else None

            # Broadcast the input data. For input tensors, we first broadcast
            # its shape and then broadcast the tensor to avoid high
            # serialization cost.
            py_data = {
                "input_tokens_size":
                input_tokens.size(),
                "input_positions_size":
                input_positions.size(),
                "is_prompt":
                input_metadata.is_prompt,
                "slot_mapping_size":
                get_size_or_none(input_metadata.slot_mapping),
                "max_context_len":
                input_metadata.max_context_len,
                "context_lens_size":
                get_size_or_none(input_metadata.context_lens),
                "block_tables_size":
                get_size_or_none(input_metadata.block_tables),
                "use_cuda_graph":
                input_metadata.use_cuda_graph,
                "selected_token_indices_size":
                sampling_metadata.selected_token_indices.size(),
            }
            broadcast_object_list([py_data], src=0)
            # TODO(zhuohan): Combine the broadcasts or set async_op=True.
            broadcast(input_tokens, src=0)
            broadcast(input_positions, src=0)
            if input_metadata.slot_mapping is not None:
                broadcast(input_metadata.slot_mapping, src=0)
            if input_metadata.context_lens is not None:
                broadcast(input_metadata.context_lens, src=0)
            if input_metadata.block_tables is not None:
                broadcast(input_metadata.block_tables, src=0)
            broadcast(sampling_metadata.selected_token_indices, src=0)
        else:
            receving_list = [None]
            broadcast_object_list(receving_list, src=0)
            py_data = receving_list[0]
            input_tokens = torch.empty(*py_data["input_tokens_size"],
                                       dtype=torch.long,
                                       device="cuda")
            broadcast(input_tokens, src=0)
            input_positions = torch.empty(*py_data["input_positions_size"],
                                          dtype=torch.long,
                                          device="cuda")
            broadcast(input_positions, src=0)
            if py_data["slot_mapping_size"] is not None:
                slot_mapping = torch.empty(*py_data["slot_mapping_size"],
                                           dtype=torch.long,
                                           device="cuda")
                broadcast(slot_mapping, src=0)
            else:
                slot_mapping = None
            if py_data["context_lens_size"] is not None:
                context_lens = torch.empty(*py_data["context_lens_size"],
                                           dtype=torch.int,
                                           device="cuda")
                broadcast(context_lens, src=0)
            else:
                context_lens = None
            if py_data["block_tables_size"] is not None:
                block_tables = torch.empty(*py_data["block_tables_size"],
                                           dtype=torch.int,
                                           device="cuda")
                broadcast(block_tables, src=0)
            else:
                block_tables = None
            selected_token_indices = torch.empty(
                *py_data["selected_token_indices_size"],
                dtype=torch.long,
                device="cuda")
            broadcast(selected_token_indices, src=0)
            input_metadata = InputMetadata(
                is_prompt=py_data["is_prompt"],
                slot_mapping=slot_mapping,
                max_context_len=py_data["max_context_len"],
                context_lens=context_lens,
                block_tables=block_tables,
                use_cuda_graph=py_data["use_cuda_graph"],
            )
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                seq_data=None,
                prompt_lens=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                perform_sampling=False,
            )

        return input_tokens, input_positions, input_metadata, sampling_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[SamplerOutput]:
        input_tokens, input_positions, input_metadata, sampling_metadata = (
            self.prepare_input_tensors(seq_group_metadata_list))
        # passing input data as well to ease process of model conversion
        patch_model_with_openvino(self.model, self.model_config,
                                    input_ids=input_tokens,
                                    positions=input_positions,
                                    kv_caches=kv_caches,
                                    input_metadata=input_metadata)
        # Execute the model.
        if input_metadata.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model
        hidden_states = model_executable(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            input_metadata=input_metadata,
        )

        # Sample the next token.
        output = self.model.sample(
            hidden_states=hidden_states,
            sampling_metadata=sampling_metadata,
        )
        return output

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model_config.get_vocab_size()
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [(None, None)] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[KVCache]) -> None:
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, 1, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, 1,
                                      dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, 1, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        context_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        # NOTE: Capturing the largest batch size first may help reduce the
        # memory usage of CUDA graph.
        for batch_size in reversed(batch_size_capture_list):
            # Create dummy input_metadata.
            input_metadata = InputMetadata(
                is_prompt=False,
                slot_mapping=slot_mapping[:batch_size],
                max_context_len=self.max_context_len_to_capture,
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
                use_cuda_graph=True,
            )

            graph_runner = CUDAGraphRunner(self.model)
            graph_runner.capture(
                input_tokens[:batch_size],
                input_positions[:batch_size],
                kv_caches,
                input_metadata,
                memory_pool=self.graph_memory_pool,
            )
            self.graph_memory_pool = graph_runner.graph.pool()
            self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info(f"Graph capturing finished in {elapsed_time:.0f} secs.")


class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        memory_pool,
    ) -> None:
        assert self.graph is None
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        self.model(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
        )
        torch.cuda.synchronize()

        # Capture the graph.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=memory_pool):
            hidden_states = self.model(
                input_ids,
                positions,
                kv_caches,
                input_metadata,
            )
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": input_metadata.slot_mapping,
            "context_lens": input_metadata.context_lens,
            "block_tables": input_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(input_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["context_lens"].copy_(input_metadata.context_lens,
                                                 non_blocking=True)
        self.input_buffers["block_tables"].copy_(input_metadata.block_tables,
                                                 non_blocking=True)

        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _pad_to_max(x: List[int], max_len: int, pad: int) -> List[int]:
    assert len(x) <= max_len
    return x + [pad] * (max_len - len(x))


def _make_tensor_with_pad(
    x: List[List[int]],
    max_len: int,
    pad: int,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,
) -> torch.Tensor:
    padded_x = [_pad_to_max(x_i, max_len, pad) for x_i in x]
    return torch.tensor(padded_x,
                        dtype=dtype,
                        device=device,
                        pin_memory=pin_memory and str(device) == "cpu")


def _get_graph_batch_size(batch_size: int) -> int:
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return (batch_size + 7) // 8 * 8


def _async_h2d(data: list, dtype: torch.dtype,
    device: Union[str, torch.device] = "cuda",
    pin_memory: bool = False,):
    t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory)
    return t.to(device=device, non_blocking=True)
