"""Regressions for real rendering, lock and matching failures; no model downloads."""
import ast
from collections import defaultdict
from pathlib import Path
import sys
import unittest
import numpy as np

VEIL=Path(__file__).resolve().parents[1]/'models/veil'


def load_functions(filename,names,namespace):
    tree=ast.parse((VEIL/filename).read_text())
    tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    exec(compile(tree,str(VEIL/filename),'exec'),namespace)
    return namespace


class ResearchCorrectnessTests(unittest.TestCase):
    def test_noop_blur_does_not_claim_pixels_changed(self):
        sys.path.insert(0,str(VEIL))
        try:
            from crop_manager import apply_fallback_blur
            flat=np.full((30,30,3),100,np.uint8)
            self.assertFalse(apply_fallback_blur(flat,[0,0,30,30],return_status=True)[1])
            self.assertFalse(apply_fallback_blur(flat,[40,40,50,50],return_status=True)[1])
            noisy=np.random.default_rng(1).integers(0,256,flat.shape,dtype=np.uint8)
            self.assertTrue(apply_fallback_blur(noisy,[0,0,30,30],return_status=True)[1])
        finally: sys.path.remove(str(VEIL))

    def test_later_background_blur_cannot_change_overlapping_target(self):
        ns=load_functions('render_safety.py',{'restore_protected_regions','changed_pixels'},{'np':np})
        original=np.random.default_rng(2).integers(0,256,(30,40,3),dtype=np.uint8)
        output=np.zeros_like(original)
        ctx={'is_target_final':True,'raw_bbox':[5,5,20,20],'smoothed_bbox':[7,4,22,20]}
        ns['restore_protected_regions'](output,original,[ctx])
        self.assertEqual(ns['changed_pixels'](original,output,ctx['raw_bbox']),0)
        self.assertEqual(ns['changed_pixels'](original,output,ctx['smoothed_bbox']),0)
        self.assertGreater(ns['changed_pixels'](original,output,[25,0,40,30]),0)

    def test_lock_expires_and_track_reuse_cannot_inherit_other_face(self):
        ns=load_functions('main_hybrid.py',{'mark_target_seen','is_known_target_face',
            'is_recent_target_track','cleanup_target_last_seen'},
            {'target_face_ids':set(),'target_track_ids':set(),'target_last_seen':{},
             'target_track_faces':{},'TARGET_HOLD_FRAMES':60})
        ns['mark_target_seen'](7,2,1)
        self.assertTrue(ns['is_known_target_face'](7,61))
        self.assertFalse(ns['is_known_target_face'](7,62))
        self.assertTrue(ns['is_recent_target_track'](2,2,7))
        self.assertFalse(ns['is_recent_target_track'](2,2,8))
        ns['cleanup_target_last_seen'](62)
        self.assertEqual(ns['target_face_ids'],set())

    def test_failed_refresh_cannot_reuse_embedding_indefinitely(self):
        ns=load_functions('face_identifier.py',{'get_track_embedding'},{
            'np':np,'track_last_emb':{2:np.ones(3)},'track_last_kps':{},
            'track_last_emb_frame':{2:1},'track_last_emb_bbox':{2:[0,0,100,100]},
            'track_to_face':{},'face_last_bbox':{},'BBOX_FALLBACK_MAX_AGE':30,
            'BBOX_FALLBACK_CENTER_RATIO':.75,'EMBEDDING_REFRESH_INTERVAL':5,
            'is_invalid_identity_bbox':lambda *args:False,
            'get_arcface_embedding':lambda *args:(None,None)})
        self.assertIsNotNone(ns['get_track_embedding'](None,[0,0,100,100],2,10)[0])
        self.assertIsNone(ns['get_track_embedding'](None,[0,0,100,100],2,35)[0])
        self.assertIsNone(ns['get_track_embedding'](None,[200,0,300,100],2,10)[0])

    def test_one_system_observation_cannot_match_two_gt_people(self):
        from tools.evaluate_veil_metadata import compute_gt_target_metrics,empty_gt_metrics
        actions=[{'frame_idx':1,'bbox':[0,0,20,20],'state':'PRESERVE'}]
        gt=[{'frame_idx':1,'bbox':[0,0,20,20],'person_id':str(i)} for i in range(2)]
        metrics,rows=compute_gt_target_metrics('a',actions,gt,iou_threshold=.5)
        self.assertEqual(metrics['gt_target_matched_boxes'],1)
        self.assertEqual(metrics['gt_target_missed_boxes'],1)
        self.assertIsNone(empty_gt_metrics(.5)['gt_target_altered_rate'])


if __name__=='__main__': unittest.main()
