import unittest

from mbcdisasm.ast import LuaReconstructor
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram
from mbcdisasm.manual_semantics import InstructionSemantics, StackEffect


def _semantics(
    key: str,
    mnemonic: str,
    *,
    summary: str,
    stack_delta: float,
    inputs: int,
    outputs: int,
    tags: tuple[str, ...],
    vm_style: str = "method",
) -> InstructionSemantics:
    return InstructionSemantics(
        key=key,
        mnemonic=mnemonic,
        manual_name=mnemonic,
        summary=summary,
        control_flow=None,
        control_flow_confidence=None,
        control_flow_reasons=(),
        stack_delta=stack_delta,
        stack_effect=StackEffect(inputs=inputs, outputs=outputs, delta=stack_delta, source="test"),
        tags=tags,
        comparison_operator=None,
        enum_values={},
        enum_namespace=None,
        struct_context=None,
        stack_inputs=inputs,
        stack_outputs=outputs,
        uses_operand=True,
        operand_hint=None,
        vm_method=mnemonic,
        vm_call_style=vm_style,
    )


class LuaReconstructorTest(unittest.TestCase):
    def test_literal_push_and_stack_updates(self) -> None:
        instructions = [
            IRInstruction(
                offset=0x10,
                key="00:00",
                mnemonic="push_literal_small",
                operand=5,
                stack_delta=1,
                control_flow="fallthrough",
                semantics=_semantics(
                    "00:00",
                    "push_literal_small",
                    summary="Push literal",
                    stack_delta=1,
                    inputs=0,
                    outputs=1,
                    tags=("literal",),
                    vm_style="literal",
                ),
                stack_inputs=0,
                stack_outputs=1,
            ),
            IRInstruction(
                offset=0x14,
                key="00:01",
                mnemonic="fold_stack_pair",
                operand=0,
                stack_delta=-2,
                control_flow=None,
                semantics=_semantics(
                    "00:01",
                    "fold_stack_pair",
                    summary="fold",
                    stack_delta=-2,
                    inputs=2,
                    outputs=0,
                    tags=("pop",),
                ),
                stack_inputs=2,
                stack_outputs=0,
            ),
        ]
        block = IRBlock(start=0x10, instructions=instructions, successors=[])
        program = IRProgram(segment_index=3, blocks={block.start: block})

        reconstructor = LuaReconstructor()
        function = reconstructor.from_ir(3, program)
        rendered = reconstructor.render(function)

        self.assertIn("local stack = {}", rendered)
        self.assertIn("local vm = {}", rendered)
        self.assertIn("vm:push_literal_small(5)", rendered)
        self.assertIn("stack[#stack + 1] = literal_0", rendered)
        self.assertIn("stack[#stack] = nil", rendered)
        self.assertIn("warnings: underflow", rendered)

        metadata = function.metadata()
        self.assertTrue(any(item["type"] == "operation" for item in metadata))
        json_payload = function.to_json(indent=0)
        self.assertIn('"type": "operation"', json_payload)
        self.assertEqual(function.operation_count(), 2)
        self.assertGreaterEqual(function.warning_count(), 1)
        summary = function.summary()
        self.assertEqual(summary["operations"], function.operation_count())
        self.assertTrue(summary["has_warnings"])
        self.assertTrue(summary["labels"])
        self.assertTrue(function.has_warnings())
        self.assertTrue(any(label.startswith("block_") for label in function.labels()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
