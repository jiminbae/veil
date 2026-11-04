import csv
import json
from pathlib import Path
import tempfile
import unittest
from tools.aggregate_research_results import aggregate,cluster_interval,read_csv
from tools.evaluate_research_run import csv_write


class ResearchStatisticsTests(unittest.TestCase):
    def test_micro_and_clip_macro_have_distinct_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            clips=[{'condition':'full','clip_id':'a','sequence':'d1_a','status':'completed',
                'target_preserve_count':90,'derived_target_observations':100},
                {'condition':'full','clip_id':'b','sequence':'d2_b','status':'completed',
                'target_preserve_count':5,'derived_target_observations':10}]
            runtime=[{'condition':'full','clip_id':c,'end_to_end_fps':10,'end_to_end_sec':10,'frames':100} for c in ['a','b']]
            csv_write(root/'per_clip_metrics.csv',clips);csv_write(root/'runtime_metrics.csv',runtime)
            aggregate(root,resamples=20)
            rows=read_csv(root/'aggregate_metrics.csv')
            values={r['aggregation']:float(r['value']) for r in rows if r['condition']=='full' and r['metric']=='derived_target_preserve_rate'}
            self.assertAlmostEqual(values['micro'],95/110)
            self.assertAlmostEqual(values['clip_macro'],.7)

    def test_cluster_count_does_not_treat_frames_as_independent_units(self):
        rows=[{'sequence':'a','v':.1} for _ in range(1000)]+[{'sequence':'b','v':.9}]
        _,_,n=cluster_interval(rows,lambda r:r['v'],resamples=20)
        self.assertEqual(n,2)
        self.assertEqual(cluster_interval(rows[:1000],lambda r:r['v'],resamples=20),(None,None,1))


if __name__=='__main__':unittest.main()
