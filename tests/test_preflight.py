import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from so101_m1 import preflight


class PreflightTests(unittest.TestCase):
    def test_missing_failed_stale_and_incomplete_evidence_block_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            with patch.object(preflight,'ARTIFACTS',root),patch.object(preflight,'ROOT',root):
                with self.assertRaisesRegex(RuntimeError,'run scripts'): preflight.require_preflight()
                report=root/'physics_validation.json'
                report.write_text(json.dumps({'contact_checks_pass':False}))
                with self.assertRaisesRegex(RuntimeError,'not passed'): preflight.require_preflight()
                source=root/'model.py'; source.write_text('validated model')
                metadata={'contact_checks_pass':True,'source_sha256':{'model.py':hashlib.sha256(source.read_bytes()).hexdigest()},'comparisons':[{'case':'native_vs_warp'}]}
                report.write_text(json.dumps(metadata))
                preflight.require_preflight(smoke=True)
                with self.assertRaisesRegex(RuntimeError,'full-path'): preflight.require_preflight()
                source.write_text('changed model')
                with self.assertRaisesRegex(RuntimeError,'stale'): preflight.require_preflight(smoke=True)


if __name__=='__main__':unittest.main()
