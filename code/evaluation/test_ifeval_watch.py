import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import watch_ifeval_recovery as watch


class WatchTests(unittest.TestCase):
    def test_live_state_and_partial_concurrent_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for model in watch.MODELS:
                d=root/model;d.mkdir();(d/'resumed').mkdir()
                (d/'recovery_state.json').write_text(json.dumps(dict(model=model,status='running',phase='resume')))
                (d/'resumed/generations.jsonl').write_text('{"key":1}\n{"partial":')
            with patch.object(watch,'ROOT',root):rows=watch.snapshot()
            self.assertEqual([r['completed'] for r in rows],[138,133,146,137])
            self.assertTrue(all(r['status']=='running' for r in rows))


if __name__=='__main__':unittest.main()
