import pytest

from mbcdisasm.adb import SegmentDescriptor, SegmentIndex
from mbcdisasm.mbc import MbcContainer


def test_segment_index_validates_offsets_are_contiguous():
    index = SegmentIndex([60, 100, 150], total_length=200)

    assert [seg.start for seg in index] == [60, 100, 150]
    assert [seg.end for seg in index] == [100, 150, 200]


def test_segment_index_handles_empty_adb_single_segment():
    index = SegmentIndex([], total_length=32)

    assert len(index) == 1
    assert index[0].start == 0
    assert index[0].end == 32


def test_segment_index_handles_empty_adb_zero_length():
    index = SegmentIndex([], total_length=0)

    assert len(index) == 0


@pytest.mark.parametrize(
    "offsets,total_length",
    [
        ([60, 50], 200),
        ([60, 120, 119], 200),
    ],
)
def test_segment_index_rejects_descending_offsets(offsets, total_length):
    with pytest.raises(ValueError, match="non-decreasing order"):
        SegmentIndex(offsets, total_length)


@pytest.mark.parametrize(
    "offsets,total_length",
    [
        ([0], 0),
        ([10], 5),
        ([0, 4, 12], 10),
    ],
)
def test_segment_index_rejects_out_of_bounds_segments(offsets, total_length):
    with pytest.raises(ValueError, match="(bounds|positive length|offsets outside)"):
        SegmentIndex(offsets, total_length)


def test_slice_segment_rejects_invalid_range():
    descriptor = SegmentDescriptor(index=0, start=5, end=12)
    mbc_data = b"abc"

    with pytest.raises(ValueError, match="boundaries"):
        MbcContainer._slice_segment(mbc_data, descriptor)
