import pytest

from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v2 import _post_moe_needs_routed_scaling
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu-intel")


def _device_flags(monkeypatch, *, cuda=False, musa=False, xpu=False, aiter=False):
    monkeypatch.setattr(deepseek_v2, "_is_cuda", cuda)
    monkeypatch.setattr(deepseek_v2, "_is_musa", musa)
    monkeypatch.setattr(deepseek_v2, "_is_xpu", xpu)
    monkeypatch.setattr(deepseek_v2, "_use_aiter", aiter)


def _kt_method() -> KTEPWrapperMethod:
    # The predicate only looks at the type; wrapper state is never touched.
    return object.__new__(KTEPWrapperMethod)


def test_kt_skips_the_scale_when_topk_folded_it(monkeypatch):
    # MXFP4 methods report fuse_routed_scaling_factor_in_topk, so the routing
    # weights already carry the factor and scaling again would square it.
    _device_flags(monkeypatch, cuda=True)
    assert not _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=True)


def test_kt_scales_once_when_topk_did_not_fold_it(monkeypatch):
    # e.g. KT over FP8: the top-k leaves the factor out, so the epilogue owes it.
    _device_flags(monkeypatch, cuda=True)
    assert _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=False)


def test_kt_ignores_device_flags(monkeypatch):
    _device_flags(monkeypatch)
    assert not _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=True)
    assert _post_moe_needs_routed_scaling(_kt_method(), fused_in_topk=False)


def test_cuda_runner_owns_the_scale(monkeypatch):
    _device_flags(monkeypatch, cuda=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=False)


def test_cpu_epilogue_scales_unless_folded(monkeypatch):
    _device_flags(monkeypatch)
    assert _post_moe_needs_routed_scaling(object(), fused_in_topk=False)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=True)


def test_musa_and_xpu_leave_the_scale_to_the_runner(monkeypatch):
    _device_flags(monkeypatch, musa=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=False)
    _device_flags(monkeypatch, xpu=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=False)


def test_aiter_folded_scale_is_not_reapplied(monkeypatch):
    _device_flags(monkeypatch, aiter=True)
    assert not _post_moe_needs_routed_scaling(object(), fused_in_topk=True)
    assert _post_moe_needs_routed_scaling(object(), fused_in_topk=False)