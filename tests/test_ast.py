import unittest

from mbcdisasm.ast import LuaReconstructor
from mbcdisasm.ir import IRBlock, IRInstruction, IRProgram


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
            ),
            IRInstruction(
                offset=0x14,
                key="00:01",
                mnemonic="fold_stack_pair",
                operand=0,
                stack_delta=-2,
                control_flow=None,
            ),
        ]
        block = IRBlock(start=0x10, instructions=instructions, successors=[])
        program = IRProgram(segment_index=3, blocks={block.start: block})

        reconstructor = LuaReconstructor()
        function = reconstructor.from_ir(3, program)
        rendered = reconstructor.render(function)

        self.assertIn("local stack = {}", rendered)
        self.assertIn("stack[#stack + 1] = 5", rendered)
        self.assertIn("stack[#stack] = nil", rendered)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
