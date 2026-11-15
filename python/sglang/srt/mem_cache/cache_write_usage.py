"""Request-local measurement of newly published, reusable prompt KV.

This is an engine observation, not a TTL, durability or billing guarantee.
It deliberately has an unknown state: an uninstrumented cache path must not
report zero as if the absence of cache writes had been established.
"""

from __future__ import annotations

from typing import Iterable, Optional


class CacheWriteUsage:
    """Union newly adopted token ranges, excluding output and internal retries.

    Ownership alone is insufficient for hybrid models: full attention KV also
    needs a reusable recurrent-state boundary. Keep the adopted ranges so a
    later successful checkpoint can make an earlier range reusable. Once a
    range has been counted, a retraction or eviction does not count it again.
    """

    __slots__ = (
        "prompt_tokens",
        "_excluded_suffix_tokens",
        "_supported",
        "_adopted",
        "_reusable",
        "_cached_prefix_len",
        "_started",
    )

    def __init__(self, prompt_tokens: int, *, excluded_suffix_tokens: int = 0):
        if (
            not isinstance(excluded_suffix_tokens, int)
            or isinstance(excluded_suffix_tokens, bool)
            or excluded_suffix_tokens < 0
        ):
            raise ValueError("excluded_suffix_tokens must be a non-negative integer")
        self.prompt_tokens = max(0, prompt_tokens)
        self._excluded_suffix_tokens = excluded_suffix_tokens
        self._supported: Optional[bool] = None
        self._adopted: list[tuple[int, int]] = []
        self._reusable: list[tuple[int, int]] = []
        self._cached_prefix_len = 0
        self._started = False

    def sync_prompt_tokens(self, prompt_tokens: int) -> None:
        """Use post-padding/truncation input length before the first insert.

        A changed prompt after publication invalidates the coordinate system
        of earlier ranges, so the amount must become unknown instead of being
        silently clipped or expanded.
        """
        prompt_tokens = max(0, prompt_tokens)
        if prompt_tokens == self.prompt_tokens:
            return
        if self._started:
            self.mark_unknown()
        self.prompt_tokens = prompt_tokens

    def set_excluded_suffix_tokens(self, excluded_suffix_tokens: int) -> None:
        if (
            not isinstance(excluded_suffix_tokens, int)
            or isinstance(excluded_suffix_tokens, bool)
            or excluded_suffix_tokens < 0
        ):
            raise ValueError("excluded_suffix_tokens must be a non-negative integer")
        if excluded_suffix_tokens == self._excluded_suffix_tokens:
            return
        if self._started:
            self.mark_unknown()
        self._excluded_suffix_tokens = excluded_suffix_tokens

    @property
    def reported_prompt_tokens(self) -> int:
        # Physical KV and checkpoint coordinates retain the complete prompt.
        # Only the public observation excludes a server-selected input suffix.
        return max(0, self.prompt_tokens - self._excluded_suffix_tokens)

    @property
    def tokens(self) -> Optional[int]:
        if self._supported is not True:
            return None
        return sum(
            max(
                0,
                min(end, self.reported_prompt_tokens)
                - max(start, self._cached_prefix_len),
            )
            for start, end in self._reusable
        )

    def mark_supported(self) -> None:
        # An unsupported path cannot become measurable halfway through a req.
        if self._supported is None:
            self._supported = True

    def mark_unknown(self) -> None:
        self._supported = False
        self._adopted.clear()
        self._reusable.clear()

    def record(
        self,
        adopted_ranges: Iterable[tuple[int, int]],
        *,
        reusable_prefix_len: int,
        cached_prefix_len: int = 0,
    ) -> None:
        if self._supported is not True:
            return
        self._started = True
        # Recomputing an originally read prefix after retraction is not a new
        # customer write. Keep read and write categories disjoint.
        self._cached_prefix_len = max(self._cached_prefix_len, cached_prefix_len)
        for start, end in adopted_ranges:
            self._merge(self._adopted, max(0, start), min(self.prompt_tokens, end))
        boundary = max(0, min(self.prompt_tokens, reusable_prefix_len))
        for start, end in self._adopted:
            self._merge(self._reusable, start, min(boundary, end))

    @staticmethod
    def _merge(ranges: list[tuple[int, int]], start: int, end: int) -> None:
        if start >= end:
            return
        first = 0
        while first < len(ranges) and ranges[first][1] < start:
            first += 1
        last = first
        while last < len(ranges) and ranges[last][0] <= end:
            start = min(start, ranges[last][0])
            end = max(end, ranges[last][1])
            last += 1
        ranges[first:last] = [(start, end)]


def cache_write_usage_for_req(req) -> CacheWriteUsage:
    """Also permit lightweight request fixtures used by cache consumers."""
    usage = getattr(req, "_cache_write_usage", None)
    if usage is None:
        usage = req._cache_write_usage = CacheWriteUsage(len(req.origin_input_ids))
    else:
        usage.sync_prompt_tokens(len(req.origin_input_ids))
    return usage


PD_SNAPSHOT_MAX_RANGES = 3
PD_SNAPSHOT_CARRIED = 1
PD_SNAPSHOT_NOT_CARRIED = 0


def export_pd_cache_write_snapshot(
    usage: "CacheWriteUsage",
) -> tuple[int, int, list[tuple[int, int]]]:
    """Bound the measured write ranges for the PD metadata carrier.

    Returns (flag, total, ranges). ``flag == 0`` means "not carried": an
    unsupported path must not masquerade as a measured zero. Exported ranges
    are already net of the prefill's own cached reads and clipped to the
    public prompt window, so the receiver needs no prefill coordinate space.
    """
    if usage is None or usage._supported is not True:
        # No meter at the send boundary: the request was never measured.
        return (PD_SNAPSHOT_NOT_CARRIED, 0, [])
    upper = usage.reported_prompt_tokens
    clipped = []
    for start, end in usage._reusable:
        start = max(start, usage._cached_prefix_len)
        end = min(end, upper)
        if start < end:
            clipped.append((start, end))
    if len(clipped) > PD_SNAPSHOT_MAX_RANGES:
        # A bounded carrier cannot represent this coverage; keep the
        # request unknown rather than publishing a partial measurement.
        return (PD_SNAPSHOT_NOT_CARRIED, 0, [])
    return (
        PD_SNAPSHOT_CARRIED,
        sum(end - start for start, end in clipped),
        clipped,
    )


def import_pd_cache_write_snapshot(
    usage: "CacheWriteUsage",
    flag: int,
    total: int,
    ranges: list[tuple[int, int]],
    prompt_tokens: int,
) -> bool:
    """Validate a transported snapshot and adopt it as this request's write.

    Returns True when the request is measured. Overlapping decode-side
    adoptions merge by union in :meth:`CacheWriteUsage.record`, so a decode
    cache that re-adopts transferred ranges cannot double count.
    """
    if flag != PD_SNAPSHOT_CARRIED:
        return False
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        return False
    if len(ranges) > PD_SNAPSHOT_MAX_RANGES:
        return False
    if not ranges and total != 0:
        return False
    checked: list[tuple[int, int]] = []
    for start, end in ranges:
        if not isinstance(start, int) or not isinstance(end, int):
            return False
        if isinstance(start, bool) or isinstance(end, bool):
            return False
        if not 0 <= start < end <= prompt_tokens:
            return False
        checked.append((start, end))
    if sum(end - start for start, end in checked) != total:
        return False
    usage.mark_supported()
    # Exported ranges are net of prefill reads; the decode-side read boundary
    # is applied by later record() calls through the existing max() fence.
    usage.record(checked, reusable_prefix_len=max(0, prompt_tokens))
    return True
