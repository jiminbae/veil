#!/usr/bin/env python3
"""Create a private browser review surface; never invent human verification."""
import argparse
import json
from pathlib import Path
import subprocess
from tools.prepare_research_inputs import sha256,write_json,ROOT

HTML=r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>VEIL private annotation review</title><style>
body{font:16px system-ui;margin:24px;background:#f4f5f7;color:#17212c;max-width:1500px}
button,input,select{font:inherit;margin:5px;padding:7px}#stage{position:relative;max-width:1200px}
video{width:100%;display:block}canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#ref{max-width:220px;max-height:180px}pre{white-space:pre-wrap}label{display:inline-block}
</style><h1>Private annotation review</h1>
<p>Official 3D projections and target mappings are <b>unverified</b>. Inspect visibility,
temporal alignment, identity and boxes. Only explicitly reviewed observations acquire a
human review record. Restricted source and output media must stay local.</p>
<select id="clips"></select><label>Reviewer <input id="reviewer" placeholder="Your reviewer ID"></label>
<div><img id="ref" alt="Reconstructed target registration crop"></div>
<div id="stage"><video id="video" controls></video><canvas id="canvas"></canvas></div>
<button id="prev">Previous frame</button><button id="next">Next frame</button>
<label>Frame <input id="frame" type="number" min="1" style="width:90px"></label>
<button id="seek">Seek</button><span id="info"></span>
<div><label>Person <select id="person"></select></label>
<label>Box x1,y1,x2,y2 <input id="bbox" size="35"></label>
<label>Visibility <select id="visibility"><option>camera_visible</option><option>occluded</option><option>not_a_face</option><option>out_of_frame</option></select></label>
<button id="review">Record this observation reviewed</button>
<button id="target">Confirm selected person as clip target</button></div>
<div><label>Missing person ID <input id="newperson"></label><label>Target <input id="newtarget" type="checkbox"></label>
<button id="add">Add missing face using entered box</button></div>
<button id="save">Download review_corrections.json</button><pre id="log"></pre>
<script>
const $=id=>document.getElementById(id);let manifest=[],labels=[],row=null,edits=[];
async function readLines(url){return (await (await fetch(url)).text()).trim().split('\n').filter(Boolean).map(JSON.parse)}
function current(){return Math.min(row.frame_count,Math.max(1,Math.floor($('video').currentTime*row.fps+1e-4)+1))}
function observation(){return labels.find(g=>g.frame_idx===current()&&g.person_id===$('person').value)}
function render(){if(!row)return;const f=current();$('frame').value=f;
 const c=$('canvas');c.width=row.width;c.height=row.height;let ctx=c.getContext('2d');
 const visible=labels.filter(g=>g.frame_idx===f);const selected=$('person').value;
 $('person').replaceChildren(...visible.map(g=>new Option(g.person_id+(g.is_target?' TARGET':''),g.person_id)));
 if(visible.some(g=>g.person_id===selected))$('person').value=selected;
 visible.forEach(g=>{let b=g.bbox;ctx.strokeStyle=g.is_target?'#ff4c4c':'#58ee8b';ctx.lineWidth=3;
 ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);ctx.font='26px sans-serif';
 ctx.fillStyle=ctx.strokeStyle;ctx.fillText(g.person_id+' '+g.visibility,b[0],Math.max(28,b[1]-8))});
 $('info').textContent='Frame '+f+'/'+row.frame_count+'; projected faces '+visible.length;
 const g=observation();$('bbox').value=g?g.bbox.join(','):'';
}
async function change(){row=manifest[$('clips').selectedIndex];
 labels=await readLines('annotations/'+row.clip_id+'.jsonl');
 $('video').src='inputs/'+row.clip_id+'/browser_review.mp4';$('ref').src='inputs/'+row.clip_id+'/target.png';render()}
function author(){const value=$('reviewer').value.trim();if(!value)throw Error('Enter your actual reviewer ID');return value}
function log(){ $('log').textContent=JSON.stringify(edits.slice(-5),null,2)+'\nTotal saved edits: '+edits.length}
$('review').onclick=()=>{try{let g=observation();if(!g)throw Error('No observation selected');
 let b=$('bbox').value.split(',').map(Number);if(b.length!==4||b.some(v=>!Number.isFinite(v))||b[2]<=b[0]||b[3]<=b[1])throw Error('Invalid box');
 edits.push({action:'review_observation',clip_id:row.clip_id,frame_idx:current(),person_id:g.person_id,
 bbox:b,visibility:$('visibility').value,reviewer:author(),reviewed_at:new Date().toISOString()});log()}catch(e){alert(e.message)}};
$('target').onclick=()=>{try{edits.push({action:'confirm_target_mapping',clip_id:row.clip_id,
 person_id:$('person').value,reviewer:author(),reviewed_at:new Date().toISOString()});log()}catch(e){alert(e.message)}};
$('add').onclick=()=>{try{const id=$('newperson').value.trim(),b=$('bbox').value.split(',').map(Number);
 if(!id||b.length!==4||b.some(v=>!Number.isFinite(v))||b[2]<=b[0]||b[3]<=b[1])throw Error('Enter a person ID and valid box');
 edits.push({action:'add_observation',clip_id:row.clip_id,frame_idx:current(),person_id:id,bbox:b,
 visibility:$('visibility').value,is_target:$('newtarget').checked,reviewer:author(),reviewed_at:new Date().toISOString()});log()}catch(e){alert(e.message)}};
$('save').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({edits},null,2)],{type:'application/json'}));a.download='review_corrections.json';a.click()};
$('next').onclick=()=>{$('video').pause();$('video').currentTime=Math.min((current())/row.fps,(row.frame_count-1)/row.fps)};
$('prev').onclick=()=>{$('video').pause();$('video').currentTime=Math.max(0,(current()-2)/row.fps)};
$('seek').onclick=()=>{$('video').currentTime=($('frame').value-1)/row.fps};
$('video').addEventListener('timeupdate',render);$('video').addEventListener('seeked',render);$('person').onchange=()=>{const g=observation();$('bbox').value=g?g.bbox.join(','):''};
$('clips').onchange=change;
readLines('inputs/dataset_manifest.jsonl').then(rows=>{manifest=rows;$('clips').replaceChildren(...rows.map(r=>new Option(r.clip_id)));return change()}).catch(e=>$('log').textContent=e);
</script></html>'''


def apply_corrections(original,corrections,output):
    edits=json.loads(Path(corrections).read_text())['edits']
    by_clip={}
    for e in edits:
        if not e.get('reviewer') or not e.get('reviewed_at'):raise ValueError('Each review needs actual reviewer and timestamp')
        by_clip.setdefault(e['clip_id'],[]).append(e)
    output=Path(output)
    if output.exists():raise ValueError('Use a fresh reviewed-annotation directory')
    output.mkdir(parents=True)
    for path in Path(original).glob('*.jsonl'):
        rows=[json.loads(v) for v in path.read_text().splitlines() if v.strip()]
        for edit in by_clip.get(path.stem,[]):
            if edit['action']=='confirm_target_mapping':
                for row in rows:
                    row['is_target']=row['person_id']==edit['person_id']
                    row['identity_mapping_review']=edit
            elif edit['action']=='review_observation':
                matches=[r for r in rows if r['frame_idx']==edit['frame_idx'] and r['person_id']==edit['person_id']]
                if len(matches)!=1:raise ValueError('Review observation must match exactly one row')
                row=matches[0];row.update(bbox=edit['bbox'],visibility=edit['visibility'],
                    verification_status='human_verified_observation',review_record=edit,
                    evaluation_eligible=edit['visibility'] not in {'not_a_face','out_of_frame'})
            elif edit['action']=='add_observation':
                if any(r['frame_idx']==edit['frame_idx'] and r['person_id']==edit['person_id'] for r in rows):
                    raise ValueError('Added face duplicates an existing person/frame key')
                rows.append({'clip_id':edit['clip_id'],'frame_idx':edit['frame_idx'],
                    'person_id':edit['person_id'],'bbox':edit['bbox'],'visibility':edit['visibility'],
                    'is_target':edit['is_target'],'annotation_source':'human_added_face',
                    'verification_status':'human_verified_observation','review_record':edit,
                    'evaluation_eligible':edit['visibility'] not in {'not_a_face','out_of_frame'}})
            else:raise ValueError('Unknown review action')
        (output/path.name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'review_audit.json').write_text(json.dumps({'edits':edits,'note':'Only individually reviewed rows are human verified'},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path)
    p.add_argument('--skip-proxies',action='store_true',help='Write the page only; proxies must already exist')
    p.add_argument('--corrections',type=Path);p.add_argument('--annotations',type=Path);p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    if a.corrections:apply_corrections(a.annotations,a.corrections,a.output_dir)
    else:
        if a.run_root is None:p.error('--run-root is required')
        (a.run_root/'annotation_review.html').write_text(HTML)
        if not a.skip_proxies:
            manifest=a.run_root/'inputs/dataset_manifest.jsonl'
            for row in (json.loads(s) for s in manifest.read_text().splitlines() if s.strip()):
                source=ROOT/row['clip_path'];dest=source.with_name('browser_review.mp4')
                receipt=dest.with_suffix('.json')
                if receipt.exists():
                    old=json.loads(receipt.read_text())
                    if old['input_sha256']==sha256(source) and dest.exists() and old['sha256']==sha256(dest):continue
                    raise ValueError('Review proxy mismatch; preserve it and use a new run')
                subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-n',
                    '-i',str(source),'-an','-c:v','libx264','-preset','fast','-crf','18',
                    '-threads','2','-movflags','+faststart',str(dest)],check=True)
                write_json(receipt,{'input_sha256':sha256(source),'sha256':sha256(dest),
                    'purpose':'browser-compatible private H264 review proxy; never used for evaluation'})


if __name__=='__main__':main()
