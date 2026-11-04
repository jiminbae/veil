"""Recover stage counts from immutable verifier receipts without rerunning models.

Masked or absent failure reasons stay unknown. FaceNet's alignment stage here
means its official detected-box crop, not five-landmark similarity alignment.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from tools.evaluate_research_run import csv_write
from tools.prepare_research_inputs import sha256, write_json


def stages(success, reason):
    if success:
        return True, True, True
    if reason == 'face_not_detected' or (reason or '').startswith('detection_error:'):
        return False, False, False
    if reason in {'alignment_failed', 'crop_failed'}:
        return True, False, False
    if reason == 'invalid_embedding':
        return True, True, False
    return None, None, False


def summarize(values):
    n = len(values)
    success = sum(v is True for v in values)
    unknown = sum(v is None for v in values)
    return dict(attempts=n, successes=success, failures=n-success-unknown,
                unknown=unknown, success_rate=success/n if n and not unknown else None,
                success_rate_lower_bound=success/n if n else None,
                success_rate_upper_bound=(success+unknown)/n if n else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--local-evaluation', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    rows, sources = [], []
    for path in sorted((a.local_evaluation/'scores').glob('*/*/scores.json')):
        records = json.loads(path.read_text())
        if not records:
            continue
        backend, clip = records[0]['backend'], records[0]['clip_id']
        receipt = a.local_evaluation/'embeddings'/backend/clip/'source_receipt.json'
        meta = json.loads(receipt.read_text())
        attempts = {r['key']: r for r in meta['attempts']}
        sources.append({'file': str(path.relative_to(a.local_evaluation)), 'sha256': sha256(path),
                        'source_receipt_sha256': sha256(receipt)})
        grouped = defaultdict(list)
        for r in records:
            source = attempts[f"f{r['frame_idx']}_p{r['person_id']}"]
            if source['success'] != r['source_embedding_ok']:
                raise RuntimeError('Source receipt and scored source embedding disagree')
            group = 'target' if r['is_target'] else 'non_target'
            for side, success, reason in [
                ('source', source['success'], source['reason']),
                ('output', r['output_embedding_ok'], r['failure_reason'])]:
                for stage, value in zip(['detection', 'alignment_or_crop', 'embedding'], stages(success, reason)):
                    grouped[side, group, stage].append(value)
        for (side, group, stage), values in sorted(grouped.items()):
            rows.append(dict(backend=backend, condition=records[0]['condition'], clip_id=clip,
                             side=side, group=group, stage=stage, **summarize(values)))
    if not rows:
        raise RuntimeError('No completed score records')
    # Reconcile exact embedding counts with the independently published table.
    import csv
    with (a.output_dir/'per_clip_verifier_metrics.csv').open() as f:
        expected = {(r['backend'], r['condition'], r['clip_id'], r['group']): r for r in csv.DictReader(f)}
    for r in rows:
        if r['stage'] != 'embedding':
            continue
        e = expected[r['backend'], r['condition'], r['clip_id'], r['group']]
        if int(e['attempts']) != r['attempts'] or int(e[r['side']+'_embedding_successes']) != r['successes']:
            raise RuntimeError('Embedding stage does not reconcile with published metrics')
    csv_write(a.output_dir/'extraction_stage_metrics.csv', rows)
    write_json(a.output_dir/'extraction_stage_status.json', dict(
        status='receipt_derived_and_embedding_counts_reconciled', rows=len(rows), sources=sources,
        reference_scope='Registered-reference extraction has only a success flag; failed reference detection/alignment stages were not logged',
        unknown_policy='Masked source_or_reference failures and unclassified failures have unknown detection/alignment; no point estimate when unknown',
        denominator='Same sampled eligible face observations as per_clip_verifier_metrics; repeated source counts across conditions are paired, not independent',
        alignment_scope='AdaFace: five-landmark similarity; FaceNet: official detected-box crop'))
    print('Wrote', len(rows), 'stage rows; exact embedding counts reconcile')


if __name__ == '__main__':
    main()
