"""Validation selection must penalize missing cases and never inspect test PSNR."""
import tempfile,unittest
from pathlib import Path
import runtime
class SelectionTest(unittest.TestCase):
 def test_coverage_before_score_and_no_test_selection(self):
  original_out,original_config=runtime.OUT,runtime.CONFIG
  with tempfile.TemporaryDirectory() as d:
   try:
    runtime.OUT=Path(d);runtime.CONFIG={'selection_uids':['a','b']}
    for variant in ['baseline','fg','fg_conf050']:
     for uid in ['a','b']:
      root=runtime.OUT/'comparison'/variant/uid
      failed=variant=='fg_conf050' and uid=='b'
      runtime.write(root/'status.json',{'state':'FAILED' if failed else 'COMPLETE'})
      score={'baseline':10,'fg':11,'fg_conf050':100}[variant]
      runtime.write(root/'gaussian/render_metrics.json',{'views':[{'split':'val','foreground_psnr':score} for _ in range(6)]+[{'split':'test','foreground_psnr':10000 if variant=='baseline' else 0}]})
    self.assertEqual(runtime.select(),'fg')
    self.assertFalse(runtime.read(runtime.OUT/'selection.json')['test_used'])
   finally:runtime.OUT, runtime.CONFIG=original_out,original_config
if __name__=='__main__':unittest.main(verbosity=2)
