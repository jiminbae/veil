import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('bash') and shutil.which('curl') and shutil.which('sha256sum'),'shell downloader tools unavailable')
class DownloadWeightsTests(unittest.TestCase):
    def test_partial_existing_checkpoint_is_preserved_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=b'complete-checkpoint-fixture'*1000
            source=root/'source.bin';source.write_bytes(data)
            weights=root/'weights';weights.mkdir()
            (weights/'yolo26x-face.pt').write_bytes(b'partial')
            sha=hashlib.sha256(data).hexdigest()
            env=dict(os.environ,VEIL_WEIGHTS_DIR=str(weights),VEIL_YOLO_URL=source.as_uri(),
                VEIL_INSWAPPER_URL=source.as_uri(),VEIL_YOLO_SHA256=sha,VEIL_INSWAPPER_SHA256=sha)
            script=Path(__file__).resolve().parents[1]/'scripts/download_weights.sh'
            subprocess.run(['bash',str(script)],env=env,check=True,capture_output=True)
            self.assertEqual((weights/'yolo26x-face.pt').read_bytes(),data)
            self.assertEqual(next(weights.glob('*.invalid.*')).read_bytes(),b'partial')
            self.assertFalse(list(weights.glob('*.part.*')))
            subprocess.run(['bash',str(script)],env=env,check=True,capture_output=True)
            self.assertEqual(len(list(weights.glob('*.invalid.*'))),1)


if __name__=='__main__':unittest.main()
