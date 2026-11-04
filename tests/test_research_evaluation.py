import json
import csv
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from tools.calibrate_independent_verifiers import calibrate_scores
from tools.evaluate_research_run import match_frame,event_stats,evaluate_matrix
from tools.eval_identity_verifier import evaluate_identity
from tools.create_annotation_review import apply_corrections


class ResearchEvaluationTests(unittest.TestCase):
    def test_failed_input_is_retained_with_unavailable_face_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.jsonl'
            manifest.write_text(json.dumps({'clip_id':'failed_a','sequence':'s1','frame_count':300})+'\n')
            benchmark=root/'benchmark';benchmark.mkdir()
            (benchmark/'run_summary.json').write_text(json.dumps([{'clip_id':'failed_a',
                'condition':'full','status':'failed','returncode':1,'elapsed_sec':2.}]))
            out=root/'out'
            evaluate_matrix(manifest,root/'annotations',benchmark,out,root/'local')
            with (out/'per_clip_metrics.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['status'],'failed')
            self.assertEqual(rows[0]['expected_frames'],'300')
            self.assertEqual(rows[0]['derived_target_preserve_rate'],'')
            self.assertFalse((out/'runtime_metrics.csv').exists())
            self.assertIsNone(json.loads((out/'integrity_checks.json').read_text())['all_run_checks_pass'])

    def test_matching_has_unique_observations_and_respects_iou_cutoff(self):
        labels=[{'bbox':[0,0,20,20]},{'bbox':[0,0,20,20]},{'bbox':[100,100,120,120]}]
        observations=[{'bbox':[0,0,20,20]},{'bbox':[500,500,520,520]}]
        result=match_frame(labels,observations)
        self.assertEqual(len(result),1)
        self.assertEqual(len({j for j,_ in result.values()}),1)

    def test_events_do_not_bridge_missing_annotation_intervals(self):
        rows=[{'person_id':'a','frame_idx':f,'state':'BLUR'} for f in [1,2,4]]
        self.assertEqual(event_stats(rows,{'BLUR'}),[('a',1,2),('a',4,4)])

    def test_calibration_ties_do_not_exceed_requested_empirical_fmr(self):
        result=calibrate_scores([.9,.8],[.1]*100,.01)
        self.assertGreater(result['threshold'],.1)
        self.assertEqual(result['empirical_fmr'],0.)

    def test_failed_verification_stays_in_attempt_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'source').mkdir();(root/'processed').mkdir()
            ref=root/'reference.png';Image.new('RGB',(20,20),'red').save(ref)
            meta=root/'actions.json';meta.write_text(json.dumps([{
                'frame_idx':1,'bbox':[0,0,20,20],'is_target':True,'stable_face_id':1,'final_action':'PRESERVE'}]))
            result=evaluate_identity(original=root/'source',processed=root/'processed',
                action_metadata=meta,protected_reference=ref,output_dir=root/'out',backend='debug_color')
            self.assertEqual(result['target_attempts'],1)
            self.assertEqual(result['target_pairs'],0)
            self.assertEqual(result['extraction_failures'],1)
            self.assertIsNone(result['target_accept_rate'])

    def test_review_only_verifies_the_explicit_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'annotations';source.mkdir()
            rows=[{'clip_id':'a','frame_idx':i,'person_id':'1','bbox':[0,0,20,20],
                   'verification_status':'unverified_projection'} for i in [1,2]]
            (source/'a.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            edits=root/'edits.json';edits.write_text(json.dumps({'edits':[{
                'action':'review_observation','clip_id':'a','frame_idx':1,'person_id':'1',
                'bbox':[0,0,20,20],'visibility':'camera_visible','reviewer':'actual_tester',
                'reviewed_at':'2026-11-04T00:00:00Z'}]}))
            apply_corrections(source,edits,root/'reviewed')
            reviewed=[json.loads(s) for s in (root/'reviewed/a.jsonl').read_text().splitlines()]
            self.assertEqual(reviewed[0]['verification_status'],'human_verified_observation')
            self.assertEqual(reviewed[1]['verification_status'],'unverified_projection')


if __name__=='__main__':unittest.main()
