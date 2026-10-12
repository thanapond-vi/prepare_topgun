"""Download source images/polygons without generating a Roboflow version.
Preserves server splits; no training, augmentation or test evaluation.
"""
import concurrent.futures
import argparse
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets' / 'roboflow_coffee_seg_source'
BASE = 'https://api.roboflow.com/bcgkku-007vy/topgun_coffee_bean_seg'
DISCARD_BOXES = False
LABELS = ['dark_roast', 'light_roast', 'medium_roast', 'unknown']
MAP = {'dark': 0, 'light': 1, 'medium': 2, 'unknow': 3, **{v:i for i,v in enumerate(LABELS)}}
KEY = dotenv_values(ROOT / '.env').get('ROBOFLOW_API_KEY')


def get_response(method, url, **kwargs):
    for attempt in range(4):
        try:
            r = requests.request(method, url, timeout=45, **kwargs)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if not r.ok:
                raise RuntimeError(f'HTTP {r.status_code}')
            return r
        except requests.RequestException:
            if attempt == 3:
                raise RuntimeError('Network request failed') from None
    raise RuntimeError('API retry limit reached')


def download(item):
    ident = item['id']
    if not ident.isalnum():
        raise ValueError('Unexpected image ID')
    record = OUT / 'records' / (ident + '.json')
    if record.exists():
        old = json.loads(record.read_text())
        if (OUT / old['image']).exists() and (OUT / old['label']).exists():
            return old
    data = get_response('GET', BASE + '/images/' + ident, params={'api_key': KEY}).json()['image']
    split = {'train':'train','valid':'val','val':'val','test':'test'}.get(data.get('split'))
    if not split:
        raise ValueError('Missing or unsupported split')
    url = data['urls']['original']
    if urlparse(url).scheme != 'https':
        raise ValueError('Non-HTTPS image URL')
    # Never forward the private API key to image storage.
    raw = get_response('GET', url).content
    ann = data.get('annotation')
    if not isinstance(ann, dict) or 'boxes' not in ann:
        raise ValueError('Missing annotations; not treated as a negative image')
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        w, h = im.size
        orientation = im.getexif().get(274, 1)
        if (int(ann['width']), int(ann['height'])) != (w, h):
            oriented = ImageOps.exif_transpose(im)
            if (int(ann['width']), int(ann['height'])) != oriented.size:
                raise ValueError('Annotation/image dimensions differ')
            im = oriented
            w, h = im.size
        fmt = im.format
        if orientation != 1:
            # Keep the raw pixel coordinate space used by the annotation dimensions.
            # Strip orientation rather than rotate pixels independently of polygons.
            clean = Image.fromarray(__import__('numpy').asarray(im.convert('RGB')))
            encoded = io.BytesIO()
            clean.save(encoded, format='PNG')
            raw = encoded.getvalue()
            fmt = 'PNG'
    ext = {'PNG':'.png','JPEG':'.jpg','WEBP':'.webp'}.get(fmt)
    if not ext:
        raise ValueError('Unsupported original image format')
    polygons=[b for b in ann['boxes'] if b.get('type')=='polygon']
    extra=[b for b in ann['boxes'] if b.get('type')!='polygon']
    ignored=[b for b in extra if b.get('type') is None
             and 0 <= float(b.get('width',-1))*float(b.get('height',-1)) < 1
             and float(b.get('width',-1)) > 0 and float(b.get('height',-1)) > 0
             and any(p['label']==b['label'] for p in polygons)]
    if DISCARD_BOXES and extra:
        if any(b.get('type') not in (None, 'box', 'rectangle') for b in extra):
            raise ValueError('Unsupported non-polygon annotation')
        if not polygons:
            raise ValueError('Excluded box-only image by user policy')
        ignored = extra
    if len(extra)!=len(ignored):
        review=OUT/'needs_review'
        review.mkdir(exist_ok=True)
        (review/(ident+ext)).write_bytes(raw)
        (review/(ident+'.json')).write_text(json.dumps({
            'id':ident,'name':data['name'],'split':data['split'],
            'annotation':{k:v for k,v in ann.items() if k!='key'}},ensure_ascii=False,indent=2))
        raise ValueError('Non-polygon annotation; original saved in needs_review')
    lines = []
    for box in polygons:
        if box.get('type') != 'polygon':
            raise ValueError('Non-polygon annotation')
        cls = MAP[box['label']]
        points = box['points']
        if len(points) < 3 or len({tuple(p) for p in points}) < 3:
            raise ValueError('Degenerate polygon')
        coords = []
        for x,y in points:
            x,y=float(x),float(y)
            if not 0 <= x <= w or not 0 <= y <= h:
                raise ValueError('Polygon coordinate outside image')
            coords.extend([x/w,y/h])
        lines.append(str(cls)+' '+' '.join(f'{v:.8f}' for v in coords))
    ip = Path('images')/split/(ident+ext)
    lp = Path('labels')/split/(ident+'.txt')
    (OUT/ip).parent.mkdir(parents=True,exist_ok=True)
    (OUT/lp).parent.mkdir(parents=True,exist_ok=True)
    (OUT/ip).write_bytes(raw)
    (OUT/lp).write_text('\n'.join(lines)+ ('\n' if lines else ''))
    result={'id':ident,'name':data['name'],'split':split,'width':w,'height':h,
            'image':str(ip),'label':str(lp),'sha256':hashlib.sha256(raw).hexdigest(),
            'annotation':{k:v for k,v in ann.items() if k != 'key'},
            'source_exif_orientation':orientation,
            'ignored_subpixel_boxes':[b['id'] for b in ignored if float(b.get('width',0))*float(b.get('height',0)) < 1],
            'discarded_box_annotations':[b.get('id') for b in extra] if DISCARD_BOXES else []}
    record.write_text(json.dumps(result,ensure_ascii=False))
    return result


def main():
    global OUT, BASE, DISCARD_BOXES
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--project', default='bcgkku-007vy/topgun_coffee_bean_seg')
    parser.add_argument('--discard-boxes', action='store_true', help='Exclude box-only images and remove boxes from mixed polygon annotations.')
    args = parser.parse_args()
    OUT = args.output.resolve()
    DISCARD_BOXES = args.discard_boxes
    BASE = 'https://api.roboflow.com/' + args.project.strip('/')
    if not KEY: raise RuntimeError('ROBOFLOW_API_KEY missing')
    (OUT/'records').mkdir(parents=True,exist_ok=True)
    index_path = OUT/'source_index.json'
    items = {x['id']:x for x in json.loads(index_path.read_text())} if index_path.exists() else {}
    total = None
    for scan in range(8):
        offset=0
        while total is None or offset < total:
            r=get_response('POST', BASE+'/search',params={'api_key':KEY},json={
                'in_dataset':True,'limit':100,'offset':offset,'fields':['id','name','split']}).json()
            total=int(r['total'])
            items.update({x['id']:x for x in r['results']})
            offset+=100
        index_path.write_text(json.dumps(list(items.values()),ensure_ascii=False,indent=2))
        print(f'Index scan {scan+1}: {len(items)}/{total}',flush=True)
        if len(items)==total:break
    if len(items)!=total:raise RuntimeError('Incomplete source index; retry later')
    print(f'Found {len(items)} dataset images',flush=True)
    (OUT/'source_index.json').write_text(json.dumps(list(items.values()),ensure_ascii=False,indent=2))
    rows=[]; errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs={pool.submit(download,i):i['id'] for i in items.values()}
        for f in concurrent.futures.as_completed(jobs):
            try:rows.append(f.result())
            except Exception as e:errors.append({'id':jobs[f],'error_type':type(e).__name__, 'reason':str(e) if isinstance(e, ValueError) else 'Request or annotation error'})
            if (len(rows)+len(errors))%100==0:print(f'Completed {len(rows)}, errors {len(errors)}',flush=True)
    excluded=[e for e in errors if e['reason']=='Excluded box-only image by user policy']
    errors=[e for e in errors if e not in excluded]
    duplicates={}
    for row in rows:duplicates.setdefault(row['sha256'],[]).append(row)
    cross=[{'ids':[v['id'] for v in vs],'splits':sorted({v['split'] for v in vs})} for vs in duplicates.values() if len({v['split'] for v in vs})>1]
    # Hash-only exclusion audit; never infer or tune on the six held-out images.
    held={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'ref').glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}}
    overlaps=[r['id'] for r in rows if r['sha256'] in held]
    summary={'project':BASE,'source_count':len(items),'downloaded':len(rows),'splits':dict(Counter(r['split'] for r in rows)),
             'classes':LABELS,'ignored_subpixel_boxes':{r['id']:r['ignored_subpixel_boxes'] for r in rows if r.get('ignored_subpixel_boxes')},'errors':errors,'exact_cross_split_duplicates':cross,'held_out_exact_matches':overlaps,
             'note':'Source pixels and polygons preserved; no version, preprocessing, augmentation or training. Exact-hash audit does not detect near duplicates.'}
    summary['excluded_box_only_images']=excluded
    summary['null_images_by_split']=dict(Counter(r['split'] for r in rows if not r['annotation']['boxes']))
    summary['discard_box_policy']=DISCARD_BOXES
    (OUT/'manifest.json').write_text(json.dumps(summary,indent=2))
    if DISCARD_BOXES:
        for row in excluded + rows:
            for file in (OUT/'needs_review').glob(row['id']+'.*'):
                file.unlink()
    if errors or cross or overlaps:
        (OUT/'data.yaml').unlink(missing_ok=True)
    if not errors and not cross and not overlaps:
        (OUT/'data.yaml').write_text('path: '+str(OUT)+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(LABELS)))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as e:
        print('Stopped safely:',type(e).__name__,flush=True)
        raise SystemExit(1)
