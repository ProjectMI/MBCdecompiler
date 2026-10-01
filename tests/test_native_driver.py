"""Small loader and installation checks; no game scripts are generated."""
from pathlib import Path
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

from test_native_planning import METADATA
from decompile.native_project import load_project
from mbc_format.common import MAGIC
from mbc_native import client_lock, install_pair


class DriverTests(unittest.TestCase):
    def test_existing_loader_and_decoder_feed_native_planner(self):
        code = bytes([79, 0, 41, 16, 7, 114])
        data = bytes(4)
        image = MAGIC + struct.pack('<4I', 0, 1, len(code), len(data)) + code + data
        image += struct.pack('<I', 1) + b'entry\0' + struct.pack('<IIBBI', 0, len(code) - 1, 255, 0, 0)
        image += struct.pack('<I', 1) + b'entry\0' + struct.pack('<III', 0, 0, 0) + METADATA
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'fixture.mbc'
            source.write_bytes(image)
            project = load_project([source], expected_modules=1)
        self.assertEqual(len(project.implementations), 1)
        self.assertEqual(project.implementations[0].function.return_type, 16)

    def test_exclusive_generation_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Path(directory)
            with client_lock(client):
                with self.assertRaises(RuntimeError):
                    with client_lock(client):
                        self.fail('A second writer acquired the same lock')
            self.assertFalse((client / '.native-generation.lock').exists())

    def test_pair_installation_rolls_back_after_second_replace_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging, client = root / 'stage', root / 'client'
            staging.mkdir()
            targets = [client / 'core/public/script/GeneratedScripts.h', client / 'core/private/script/GeneratedScripts.cpp']
            for target in targets:
                target.parent.mkdir(parents=True)
                target.write_text('previous ' + target.suffix)
                (staging / target.name).write_text('replacement ' + target.suffix)
            replace = os.replace
            calls = 0
            def failing_replace(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError('simulated installation failure')
                return replace(source, target)
            with patch('mbc_native.os.replace', side_effect=failing_replace):
                with self.assertRaisesRegex(OSError, 'simulated'):
                    install_pair(staging, client)
            self.assertEqual([target.read_text() for target in targets], ['previous .h', 'previous .cpp'])


if __name__ == '__main__':
    unittest.main()
