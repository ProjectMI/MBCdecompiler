from mbcdisasm.control_flow_summary import (
    ControlFlowMetrics,
    render_control_flow_summary,
    summarise_control_flow,
)
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram


def _block(start, successors):
    return IRBlock(start=start, instructions=[], successors=successors)


def test_control_flow_metrics_basic():
    blocks = {
        0x0000: _block(0x0000, [0x0010, 0x0020]),
        0x0010: _block(0x0010, [0x0030]),
        0x0020: _block(0x0020, [0x0030]),
        0x0030: _block(0x0030, [0x0010]),
        0x0040: _block(0x0040, []),
    }
    program = IRProgram(segment_index=0, blocks=blocks)

    metrics = summarise_control_flow(program)
    assert metrics.block_count == 5
    assert metrics.entry_points and metrics.entry_points[0] == 0x0000
    assert 0x0010 in metrics.loop_headers
    assert 0x0040 in metrics.unreachable
    assert metrics.reachable_blocks == 4
    assert metrics.exit_blocks == []
    assert metrics.branch_blocks == [0x0000]
    assert 0x0010 in metrics.merge_blocks
    assert metrics.critical_edges == [(0x0000, 0x0010)]
    assert metrics.max_successors == 2
    assert metrics.max_predecessors >= 1
    assert metrics.cyclomatic_complexity == 3
    assert metrics.connected_components == 1
    assert metrics.edge_count == 5
    assert metrics.average_successors > 1.0

    summary = render_control_flow_summary(metrics)
    assert summary[0] == "control flow summary:"
    assert any("loop headers" in line for line in summary)
    assert any("unreachable" in line for line in summary)
    assert any(line.startswith("block edges:") for line in summary)
    assert any(line.startswith("branch points:") for line in summary)
    assert any(line.startswith("merge points:") for line in summary)
    assert any(line.startswith("critical edges:") for line in summary)


def test_control_flow_metrics_to_dict():
    metrics = ControlFlowMetrics(
        block_count=3,
        entry_points=[0x0000],
        loop_headers=[0x0010],
        unreachable=[0x0020],
        reachable_blocks=2,
        exit_blocks=[0x0020],
        max_successors=2,
        max_predecessors=1,
        branch_blocks=[0x0000],
        merge_blocks=[0x0010],
        critical_edges=[(0x0000, 0x0010)],
        cyclomatic_complexity=1,
        connected_components=1,
        average_successors=1.5,
        edge_count=3,
    )
    payload = metrics.to_dict()
    assert payload["entry_points"] == ["0x000000"]
    assert payload["loop_headers"] == ["0x000010"]
    assert payload["unreachable"] == ["0x000020"]
    assert payload["exit_blocks"] == ["0x000020"]
    assert payload["branch_blocks"] == ["0x000000"]
    assert payload["critical_edges"] == [
        {"from": "0x000000", "to": "0x000010"}
    ]
    assert "blocks" in payload
