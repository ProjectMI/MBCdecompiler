import json
from pathlib import Path


def test_top_missing_opcode_names():
    """Report the top opcodes without a name annotation by stack samples."""

    opcode_profiles_path = Path(__file__).resolve().parent.parent / "knowledge" / "opcode_profiles.json"
    with opcode_profiles_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    annotations = data["annotations"]
    missing_name_entries = [
        (
            opcode,
            annotation.get("stack_samples", 0) or 0,
        )
        for opcode, annotation in annotations.items()
        if not annotation.get("name")
    ]

    # Sort from the highest to the lowest number of samples and keep the top 100 entries.
    missing_name_entries.sort(key=lambda item: item[1], reverse=True)
    top_missing = missing_name_entries[:100]

    assert len(top_missing) == min(100, len(missing_name_entries))

    for opcode, stack_samples in top_missing:
        print(f"{opcode}: stack_samples={stack_samples}")

    # The entries we printed should be sorted from highest to lowest stack sample counts.
    assert all(
        top_missing[idx][1] >= top_missing[idx + 1][1]
        for idx in range(len(top_missing) - 1)
    )
