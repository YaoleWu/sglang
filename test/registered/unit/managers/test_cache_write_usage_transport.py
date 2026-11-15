"""Cache admission accounting must survive IPC, routing, and native metadata.

These checks use the real producer/consumer methods without loading a model or
allocating GPU memory. Unknown accounting stays absent, rather than becoming a
billable-looking zero, and stream chunks carry cumulative snapshots.
"""

import asyncio
import copy
import pickle
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
from pydantic import TypeAdapter

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode  # noqa: E402
from sglang.srt.managers.detokenizer_manager import DetokenizerManager  # noqa: E402
from sglang.srt.managers.incremental_detokenize import (  # noqa: E402
    DetokenizeHeader,
    DetokenizeStateTable,
)
from sglang.srt.managers.io_struct import BatchStrOutput  # noqa: E402
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    GenerateReqInput,
    RequestOutputStats,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.multi_tokenizer_mixin import (  # noqa: E402
    _handle_output_by_index,
    _handle_output_by_indices,
)
from sglang.srt.managers.scheduler_components.output_streamer import (  # noqa: E402
    OutputSink,
    _GenerationStreamAccumulator,
)
from sglang.srt.managers.tokenizer_manager import ReqState  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats  # noqa: E402
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.utils.weight_versions import WeightVersionSpan  # noqa: E402
from sglang.srt.disaggregation.utils import MetadataBuffers  # noqa: E402
from sglang.srt.mem_cache.cache_write_usage import (  # noqa: E402
    CacheWriteUsage,
    export_pd_cache_write_snapshot,
    import_pd_cache_write_snapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(cls, values):
    count = len(values)
    fields = {
        f.name: None
        for f in msgspec.structs.fields(cls)
        if f.default is msgspec.NODEFAULT and f.default_factory is msgspec.NODEFAULT
    }
    fields.update(
        rids=[f"cache-{i}" for i in range(count)],
        http_worker_ipcs=[None] * count,
        finished_reasons=[None] * count,
        output_ids_flat=array("q", [1] * count),
        output_ids_offsets=list(range(count + 1)),
        reasoning_tokens=[0] * count,
        completion_tokens=[1] * count,
        cache_write_tokens=values,
        request_stats=[
            RequestOutputStats(
                prompt_tokens=20,
                cached_tokens=4,
                cached_tokens_details={"device": 2, "host": 2},
                retraction_count=0,
                image_tokens=0,
                audio_tokens=0,
                video_tokens=0,
                time_stats=None,
            )
            for _ in values
        ],
    )
    if cls is BatchTokenIDOutput:
        fields.update(
            detokenize_headers=[
                DetokenizeHeader(
                    prompt_tail_ids=array("q"),
                    skip_special_tokens=True,
                    spaces_between_special_tokens=True,
                    no_stop_trim=False,
                    http_worker_ipc=None,
                )
            ]
            * count,
        )
    else:
        fields["output_strs"] = ["hello"] * count
    return cls(**fields)


def _detokenize(payload):
    manager = DetokenizerManager.__new__(DetokenizerManager)
    manager.vocab_size = None
    manager.detokenize_state_table = DetokenizeStateTable(capacity=16)
    manager._decode_batch_token_id_output = lambda obj, states: (
        ["hello"] * len(obj.rids)
    )
    return manager.handle_batch_token_id_out(payload)


class CacheWriteTransportTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu"), role="tokenizer")


class TestCacheWriteTransport(CacheWriteTransportTestCase):
    def test_unknown_zero_and_positive_survive_real_transport(self):
        token_output = _batch(BatchTokenIDOutput, [None, 0, 16])
        token_output = msgpack_decode(msgpack_encode(token_output))
        str_output = _detokenize(token_output)
        for payload in (token_output, str_output):
            with self.subTest(payload=type(payload).__name__):
                received = msgpack_decode(msgpack_encode(payload))
                self.assertEqual(received.cache_write_tokens, [None, 0, 16])
                grouped = _handle_output_by_indices(received, [2, 0, 1])
                grouped = msgpack_decode(msgpack_encode(grouped))
                self.assertEqual(grouped.rids, ["cache-2", "cache-0", "cache-1"])
                self.assertEqual(grouped.cache_write_tokens, [16, None, 0])
                for index, value in enumerate([None, 0, 16]):
                    routed = _handle_output_by_index(received, index)
                    routed = msgpack_decode(msgpack_encode(routed))
                    self.assertEqual(routed.rids, [f"cache-{index}"])
                    self.assertEqual(routed.cache_write_tokens, [value])

    def test_old_ipc_without_tail_field_decodes_as_unknown(self):
        for cls in (BatchTokenIDOutput, BatchStrOutput):
            with self.subTest(payload=cls.__name__):
                self.assertEqual(cls.__struct_fields__[-1], "cache_write_tokens")
                wire = msgspec.msgpack.decode(msgpack_encode(_batch(cls, [16])))
                wire.pop()  # same-base IPC sender without the appended cache field
                decoded = msgpack_decode(msgspec.msgpack.encode(wire))
                self.assertIsNone(decoded.cache_write_tokens)
                self.assertIsNone(
                    _handle_output_by_index(decoded, 0).cache_write_tokens
                )

    def test_scheduler_reports_cumulative_snapshots(self):
        req = SimpleNamespace(
            rid="cache-stream",
            http_worker_ipc=None,
            finished=lambda: False,
            beam_group=None,
            stream=True,
            sampling_params=SimpleNamespace(
                stream_interval=1,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
                no_stop_trim=False,
            ),
            output_ids=[1],
            output_ids_through_stop=[1],
            send_token_offset=0,
            send_output_token_logprobs_offset=0,
            send_decode_id_offset=0,
            origin_input_ids=[9] * 20,
            origin_input_ids_unpadded=array("q", [9] * 20),
            finished_reason=None,
            decoded_text="",
            reasoning_tokens=0,
            cached_tokens=4,
            cache_write_tokens=8,
            get_mm_token_counts=lambda: (0, 0, 0),
            take_request_stats_due=lambda: True,
            mm_image_tokens=0,
            mm_audio_tokens=0,
            mm_video_tokens=0,
            multimodal_inputs=None,
            retraction_count=0,
            time_stats=None,
            customized_info=None,
            check_match_stop_str_prefix=lambda: False,
            take_detokenize_header_due=Mock(side_effect=[True, False, False]),
        )
        snapshots = []
        for value in (8, 16, 16):
            req.cache_write_tokens = value
            req.output_ids.append(2)
            req.output_ids_through_stop = req.output_ids
            acc = _GenerationStreamAccumulator(
                return_logprob=False,
                return_hidden_states=False,
                return_routed_experts=False,
                return_indexer_topk=False,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                disaggregation_mode=DisaggregationMode.NULL,
                default_stream_interval=1,
                default_force_stream_interval=1,
                build_request_stats=lambda req: RequestOutputStats(
                    prompt_tokens=len(req.origin_input_ids),
                    cached_tokens=req.cached_tokens,
                    cached_tokens_details={"device": 2, "host": 2},
                    retraction_count=req.retraction_count,
                    image_tokens=0,
                    audio_tokens=0,
                    video_tokens=0,
                    time_stats=None,
                ),
                is_request_level_customized_info_key=lambda key: False,
                current_weight_version=None,
                output_sink=OutputSink.DETOKENIZER,
            )
            acc.accept(req=req)
            snapshots.append(acc.to_payload(dp_rank=0, is_idle_batch=False))
        self.assertEqual([x.cache_write_tokens for x in snapshots], [[8], [16], [16]])
        self.assertEqual(
            [x.request_stats[0].cached_tokens for x in snapshots], [4, 4, 4]
        )


class TestCacheWritePDTransport(CacheWriteTransportTestCase):
    """The prefill snapshot must survive the real metadata carrier (#48300).

    The carrier is the spare slot range of the existing cached_tokens buffer,
    following the same packing convention as the multimodal token counts.
    """

    @staticmethod
    def _prefill_usage(prompt_tokens, adopted):
        usage = CacheWriteUsage(prompt_tokens)
        usage.mark_supported()
        for start, end in adopted:
            # Disjoint, non-adjacent ranges: the meter merges adjacent ones,
            # so each chunk here stays a separate carrier entry.
            usage.record([(start, end)], reusable_prefix_len=prompt_tokens)
        return usage

    def test_prefill_snapshot_round_trip_through_slot_encoding(self):
        import torch

        adopted = [(0, 24), (30, 46), (50, 58)]
        prefill = self._prefill_usage(64, adopted)
        self.assertEqual(prefill.tokens, 48)
        flag, total, ranges = export_pd_cache_write_snapshot(prefill)
        self.assertEqual(flag, 1)
        self.assertEqual(total, 48)
        self.assertEqual(ranges, adopted)

        buffers = MetadataBuffers(
            size=4,
            hidden_size=8,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=4,
        )
        req = SimpleNamespace(
            metadata_buffer_index=2,
            output_ids=[7],
            origin_input_ids=list(range(64)),
            cached_tokens=5,
            cached_tokens_device=5,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            get_mm_token_counts=lambda: (0, 0, 0),
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=None,
            bootstrap_room=None,
            customized_info=None,
        )
        req._cache_write_usage = prefill
        buffers.set_buf(req)
        row = buffers.cached_tokens[2]
        self.assertEqual(int(row[7]), 1)
        self.assertEqual(int(row[8]), 48)
        self.assertEqual(
            [
                (int(row[9]), int(row[10])),
                (int(row[11]), int(row[12])),
                (int(row[13]), int(row[14])),
            ],
            ranges,
        )

        # The decode side reads the same row shape the commit path uses.
        decode_usage = CacheWriteUsage(64)
        self.assertTrue(
            import_pd_cache_write_snapshot(
                decode_usage,
                int(row[7].item()),
                int(row[8].item()),
                [
                    (int(row[9 + 2 * i].item()), int(row[10 + 2 * i].item()))
                    for i in range(3)
                ],
                64,
            )
        )
        self.assertEqual(decode_usage.tokens, prefill.tokens)

    def test_padded_carrier_slots_do_not_reject_partial_coverage(self):
        import torch

        prefill = self._prefill_usage(64, [(10, 28), (40, 58)])
        flag, total, ranges = export_pd_cache_write_snapshot(prefill)
        self.assertEqual((flag, total, ranges), (1, 36, [(10, 28), (40, 58)]))

        buffers = MetadataBuffers(
            size=2,
            hidden_size=8,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=4,
        )
        req = SimpleNamespace(
            metadata_buffer_index=1,
            output_ids=[7],
            origin_input_ids=list(range(64)),
            cached_tokens=0,
            cached_tokens_device=0,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            get_mm_token_counts=lambda: (0, 0, 0),
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=None,
            bootstrap_room=None,
            customized_info=None,
        )
        req._cache_write_usage = prefill
        buffers.set_buf(req)
        row = buffers.cached_tokens[1]
        # The receiver reads all carrier slots; the padded (0, 0) pair must be
        # dropped before validation instead of rejecting the whole snapshot.
        parsed = []
        for i in range(3):
            start = int(row[9 + 2 * i].item())
            end = int(row[10 + 2 * i].item())
            if start < end:
                parsed.append((start, end))
        self.assertEqual(parsed, ranges)
        decode_usage = CacheWriteUsage(64)
        self.assertTrue(
            import_pd_cache_write_snapshot(
                decode_usage, int(row[7].item()), int(row[8].item()), parsed, 64
            )
        )
        self.assertEqual(decode_usage.tokens, 36)

    def test_decode_re_adopting_transferred_ranges_cannot_double_count(self):
        adopted = [(0, 24), (30, 46)]
        prefill = self._prefill_usage(64, adopted)
        flag, total, ranges = export_pd_cache_write_snapshot(prefill)
        decode_usage = CacheWriteUsage(64)
        self.assertTrue(
            import_pd_cache_write_snapshot(decode_usage, flag, total, ranges, 64)
        )
        # A decode radix cache that re-adopts exactly the transferred
        # coverage unions into the existing ranges instead of summing again.
        decode_usage.record(adopted, reusable_prefix_len=64)
        self.assertEqual(decode_usage.tokens, 40)

    def test_unsupported_and_oversized_snapshots_stay_not_carried(self):
        unsupported = CacheWriteUsage(64)
        unsupported.mark_unknown()
        self.assertEqual(export_pd_cache_write_snapshot(unsupported), (0, 0, []))
        never_started = CacheWriteUsage(64)
        self.assertEqual(export_pd_cache_write_snapshot(never_started), (0, 0, []))
        wide = self._prefill_usage(400, [(0, 50), (100, 150), (200, 250), (300, 350)])
        self.assertEqual(export_pd_cache_write_snapshot(wide), (0, 0, []))

    def test_invalid_carried_snapshots_are_rejected_as_unknown(self):
        cases = [
            (0, 0, []),  # legacy prefill: not carried
            (1, 5, [(0, 4)]),  # total disagrees with ranges
            (1, 20, [(60, 80)]),  # pair ends past the prompt window
            (1, 4, [(4, 4)]),  # empty pair
            (1, 4, [(8, 4)]),  # inverted pair
        ]
        for flag, total, ranges in cases:
            with self.subTest(flag=flag, total=total, ranges=ranges):
                usage = CacheWriteUsage(64)
                self.assertFalse(
                    import_pd_cache_write_snapshot(usage, flag, total, ranges, 64)
                )
                self.assertIsNone(usage.tokens)

    def test_warm_duplicate_and_measured_zero_survive_the_handoff(self):
        warm = self._prefill_usage(64, [(0, 16)])
        warm.record([(0, 16)], reusable_prefix_len=64, cached_prefix_len=16)
        self.assertEqual(warm.tokens, 0)
        flag, total, ranges = export_pd_cache_write_snapshot(warm)
        self.assertEqual((flag, total, ranges), (1, 0, []))
        decode_usage = CacheWriteUsage(64)
        self.assertTrue(
            import_pd_cache_write_snapshot(decode_usage, flag, total, ranges, 64)
        )
        self.assertEqual(decode_usage.tokens, 0)


class TestCacheWriteNativeMetadata(
    unittest.IsolatedAsyncioTestCase, CacheWriteTransportTestCase
):
    async def test_native_metadata_omits_unknown_and_keeps_measured_zero(self):
        for cls in (BatchTokenIDOutput, BatchStrOutput):
            for value in (None, 0, 16):
                with self.subTest(payload=cls.__name__, value=value):
                    manager = TokenizerManager.__new__(TokenizerManager)
                    manager.enable_metrics = False
                    manager.incremental_streaming_output = True
                    manager.config_value = lambda key: "test"
                    manager.dump_requests_folder = ""
                    manager.crash_dump_folder = ""
                    state = ReqState(
                        out_list=[],
                        finished=False,
                        event=asyncio.Event(),
                        obj=SimpleNamespace(stream=True, return_logprob=False),
                        time_stats=SimpleNamespace(first_token_time=1.0),
                    )
                    manager.rid_to_state = {"cache-0": state}
                    with patch(
                        "sglang.srt.managers.tokenizer_manager.get_serving",
                        return_value=SimpleNamespace(batch_notify_size=1),
                    ):
                        payload = _batch(cls, [value])
                        payload.prefill_weight_versions = [
                            [WeightVersionSpan(version="prefill-v1", start=0, end=20)]
                        ]
                        await manager._handle_batch_output(payload)
                    meta = state.out_list[-1]["meta_info"]
                    self.assertEqual(
                        meta["prefill_weight_versions"],
                        [{"version": "prefill-v1", "start": 0, "end": 20}],
                    )
                    if value is None:
                        self.assertNotIn("cache_write_tokens", meta)
                    else:
                        self.assertEqual(meta["cache_write_tokens"], value)

    async def test_cached_stats_stay_physical_while_metadata_uses_public_window(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.enable_metrics = False
        manager.incremental_streaming_output = True
        manager.config_value = lambda key: "test"
        manager.dump_requests_folder = ""
        manager.crash_dump_folder = ""
        obj = GenerateReqInput(input_ids=list(range(20)), stream=True)
        obj._cache_write_reporting_suffix_tokens = 3
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            time_stats=SimpleNamespace(first_token_time=1.0),
        )
        manager.rid_to_state = {"cache-0": state}
        stats = RequestOutputStats(
            prompt_tokens=20,
            cached_tokens=19,
            cached_tokens_details={"device": 19, "host": 0},
            retraction_count=0,
            image_tokens=0,
            audio_tokens=0,
            video_tokens=0,
            time_stats=None,
        )
        for index, value in enumerate((0, 1, None)):
            payload = _batch(BatchStrOutput, [value])
            payload.request_stats = [stats] if index != 1 else None
            with patch(
                "sglang.srt.managers.tokenizer_manager.get_serving",
                return_value=SimpleNamespace(batch_notify_size=1),
            ):
                await manager._handle_batch_output(payload)
            meta = state.out_list[-1]["meta_info"]
            self.assertEqual(meta["prompt_tokens"], 20)
            self.assertEqual(meta["cached_tokens"], 17)
            self.assertIsNone(meta["cached_tokens_details"])
            if value is None:
                self.assertNotIn("cache_write_tokens", meta)
            else:
                self.assertEqual(meta["cache_write_tokens"], value)
            self.assertEqual(state.request_stats.cached_tokens, 19)
            self.assertEqual(
                state.request_stats.cached_tokens_details, {"device": 19, "host": 0}
            )


class TestCacheWriteReportingWindowTransport(CacheWriteTransportTestCase):
    def test_native_json_cannot_select_reporting_window(self):
        adapter = TypeAdapter(GenerateReqInput)
        name = "_cache_write_reporting_suffix_tokens"
        native = adapter.validate_python({"input_ids": [1, 2, 3], name: 3})
        self.assertEqual(native._cache_write_reporting_suffix_tokens, 0)
        self.assertNotIn(name, adapter.json_schema()["properties"])
        with self.assertRaises(TypeError):
            GenerateReqInput(input_ids=[1, 2, 3], **{name: 3})

    def test_batch_and_parallel_copies_keep_only_server_selected_policy(self):
        batch = GenerateReqInput(
            input_ids=[[1, 2, 3, 4], [5, 6, 7, 8]], sampling_params={"n": 2}
        )
        batch._cache_write_reporting_suffix_tokens = 3
        batch.normalize_batch_and_arguments()
        for i in range(len(batch.input_ids)):
            item = batch[i]
            self.assertIs(batch[i], item)
            for copied in (item, copy.copy(item), pickle.loads(pickle.dumps(item))):
                self.assertEqual(copied._cache_write_reporting_suffix_tokens, 3)
        self.assertEqual(GenerateReqInput()._cache_write_reporting_suffix_tokens, 0)

    def test_policy_crosses_tokenizer_and_append_only_ipc(self):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3, 4],
            sampling_params={"temperature": 0, "max_new_tokens": 1},
        )
        obj._cache_write_reporting_suffix_tokens = 3
        obj.normalize_batch_and_arguments()
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.preferred_sampling_params = None
        manager.sampling_params_class = SamplingParams
        manager.tokenized_generate_req_type = TokenizedGenerateReqInput
        manager.tokenizer = None
        manager.model_config = SimpleNamespace(vocab_size=100)
        manager.rid_to_state = {
            obj.rid: SimpleNamespace(time_stats=APIServerReqTimeStats())
        }
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg",
            return_value=SimpleNamespace(disaggregation_transfer_backend="none"),
        ):
            tokenized = manager._create_tokenized_object(obj, None, obj.input_ids)
        self.assertEqual(tokenized.cache_write_reporting_suffix_tokens, 3)
        with patch("sglang.srt.managers.io_struct._USE_PICKLE_IPC", False):
            tokenized.wrap_pickle_fields()
            encoded = msgpack_encode(tokenized)
        self.assertEqual(msgpack_decode(encoded).cache_write_reporting_suffix_tokens, 3)
        self.assertEqual(
            pickle.loads(pickle.dumps(tokenized)).cache_write_reporting_suffix_tokens, 3
        )
        self.assertEqual(
            TokenizedGenerateReqInput.__struct_fields__[-1],
            "cache_write_reporting_suffix_tokens",
        )
        legacy = msgspec.msgpack.decode(encoded)
        legacy.pop()
        self.assertEqual(
            msgpack_decode(
                msgspec.msgpack.encode(legacy)
            ).cache_write_reporting_suffix_tokens,
            0,
        )

    def test_responses_constructs_requests_with_the_same_reporting_policy(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesRequest
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        async def run(encoding, stream, uses_response_template):
            server = OpenAIServingResponses.__new__(OpenAIServingResponses)
            server.chat_encoding_spec = encoding
            # main routes the K3 generation-stub adjustment through
            # ChatEncodingSpec.uncounted_prompt_tokens instead of an
            # attribute on the serving object. Mirror that contract.
            server._encoding_adapter = SimpleNamespace(
                uncounted_prompt_tokens=3 if encoding == "kimi_k3" else 0
            )
            server.tokenizer_manager = SimpleNamespace(
                tokenizer=SimpleNamespace(decode=lambda *a, **k: "rendered prefix")
            )
            server.enable_response_store = False
            server._translates_tools = False
            server.use_harmony = False
            server.tool_server = None
            server.tool_call_parser = None
            server.default_sampling_params = {}
            server._validate_model = lambda model: None
            server.extract_routed_dp_rank_from_header = lambda raw, rank: rank
            processed = SimpleNamespace(
                stop=None,
                tool_call_constraint=None,
                reasoning_end_token_ids=None,
                skip_special_tokens=True,
                image_data=None,
                video_data=None,
                audio_data=None,
                modalities=None,
                require_reasoning=False,
                uses_response_template=uses_response_template,
                template_tool_call_grammar=False,
                message_start_tool_call_constraint=None,
            )

            async def make_request(*args, **kwargs):
                return [], [list(range(512))], [list(range(512))], processed

            captured = []

            def generate(request_id, prompt, adapted, *args, **kwargs):
                captured.append(adapted)

                async def empty():
                    if False:
                        yield None

                return empty()

            async def full(*args, **kwargs):
                return "full"

            server._make_request = make_request
            server._generate_with_builtin_tools = generate
            server.responses_stream_generator_non_harmony = lambda *a, **k: "stream"
            server.responses_full_generator = full
            request = ResponsesRequest(
                model="test", input="hello", stream=stream, store=False
            )
            result = await server.create_responses(request)
            self.assertEqual(result, "stream" if stream else "full")
            self.assertEqual(len(captured), 1)
            self.assertEqual(len(captured[0].input_ids), 512)
            self.assertEqual(
                captured[0]._cache_write_reporting_suffix_tokens,
                3 if encoding == "kimi_k3" else 0,
            )
            if uses_response_template:
                self.assertEqual(request._response_parser_prefix, "rendered prefix")

        for encoding in ("kimi_k3", None):
            for stream in (False, True):
                for uses_response_template in (False, True):
                    with self.subTest(
                        encoding=encoding,
                        stream=stream,
                        uses_response_template=uses_response_template,
                    ):
                        asyncio.run(run(encoding, stream, uses_response_template))

    def test_responses_tool_retry_keeps_the_server_reporting_policy(self):
        from sglang.srt.entrypoints.openai.serving_responses import (
            OpenAIServingResponses,
        )

        async def run(encoding):
            server = OpenAIServingResponses.__new__(OpenAIServingResponses)
            server.chat_encoding_spec = encoding
            # main routes the K3 generation-stub adjustment through
            # ChatEncodingSpec.uncounted_prompt_tokens instead of an
            # attribute on the serving object. Mirror that contract.
            server._encoding_adapter = SimpleNamespace(
                uncounted_prompt_tokens=3 if encoding == "kimi_k3" else 0
            )
            server.is_disaggregated = False
            captured = []

            async def generate(adapted, raw):
                captured.append(adapted)
                yield {"text": "tool turn"}

            async def call_tool():
                return {"text": "tool result"}

            server.tokenizer_manager = SimpleNamespace(
                generate_request=generate,
                model_config=SimpleNamespace(context_len=4096),
                num_reserved_tokens=1,
            )
            context = SimpleNamespace(
                append_output=lambda output: None,
                need_builtin_tool_call=lambda: len(captured) == 1,
                call_tool=call_tool,
                render_for_completion=lambda: list(range(512)),
            )
            initial = GenerateReqInput(input_ids=list(range(512)))
            server._set_cache_reporting_window(initial)
            async for _ in server._generate_with_builtin_tools(
                "test", initial.input_ids, initial, {"max_new_tokens": 1}, context
            ):
                pass
            self.assertEqual(len(captured), 2)
            self.assertEqual(
                [item._cache_write_reporting_suffix_tokens for item in captured],
                [3 if encoding == "kimi_k3" else 0] * 2,
            )

        for encoding in ("kimi_k3", None):
            with self.subTest(encoding=encoding):
                asyncio.run(run(encoding))


if __name__ == "__main__":
    unittest.main()
