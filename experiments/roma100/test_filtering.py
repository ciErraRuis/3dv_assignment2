import unittest,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from filtering import FilterPolicy,BorderWhiteForeground,inside,FilteredMatcher
from reconlab.benchmark.core import PairMatches
class Tests(unittest.TestCase):
 def test_boundary_white_removed_enclosed_white_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   a=np.full((32,32,3),255,np.uint8);a[8:24,8:24]=100;a[12:20,12:20]=255;p=Path(d)/'x.png';Image.fromarray(a).save(p)
   mask=BorderWhiteForeground(FilterPolicy())(p)
   self.assertFalse(mask[0,0]);self.assertTrue(mask[16,16]);self.assertTrue(mask[8,8]);self.assertFalse(inside(np.array([[-2,3],[40,20]]),mask).any())
 def test_score_filter_preserves_correspondence_alignment_no_fallback(self):
  class Raw:
   def prepare(self,p):pass
   def match(self,a,b):return PairMatches([[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]],[.2,.5,.9])
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a.png';b=Path(d)/'b.png';f=FilteredMatcher(Raw(),FilterPolicy(foreground=False,minimum_overlap=.5));f.evidence_root=Path(d)/'evidence';f.prepare([a,b]);m=f.match(a,b)
   np.testing.assert_array_equal(m.first,[[2,2],[3,3]]);np.testing.assert_array_equal(m.second,[[5,5],[6,6]])
   f.policy=FilterPolicy(foreground=False,minimum_overlap=1.);self.assertEqual(len(f.match(a,b).scores),0)
if __name__=='__main__':unittest.main(verbosity=2)
