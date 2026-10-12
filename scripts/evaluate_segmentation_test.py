"""Evaluate a fixed checkpoint on test once; never tune or deploy."""
import os,json,hashlib,csv,argparse
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'runs/.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 import torch,yaml
 from ultralytics import YOLO,settings
 settings.update({'sync':False});torch.set_num_threads(4)
 parser=argparse.ArgumentParser()
 parser.add_argument('--run',type=Path)
 args=parser.parse_args()
 run=args.run or Path((ROOT/'runs/segment/latest_run.txt').read_text().strip())
 color=json.loads((run/'experiment.json').read_text()).get('color_order','rgb') if (run/'experiment.json').exists() else 'rgb'
 assert json.loads((run/'status.json').read_text())['state']=='completed'
 weights=run/'weights/best.pt';data=ROOT/'datasets/roboflow_coffee_seg_source/data.yaml'
 out=ROOT/'reports'/('test-'+run.name);out.mkdir(exist_ok=True)
 assert not (out/'results.json').exists(),'Test already evaluated; read saved report'
 config=yaml.safe_load(data.read_text());ds=Path(config['path']);names=config['names'];frozen=sha(weights)
 files=sorted(p for p in (ds/config['test']).glob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
 audit=json.loads((run/'dataset_audit.json').read_text()) if (run/'dataset_audit.json').exists() else json.loads((ds/'manifest.json').read_text())
 assert len(files)==audit['splits']['test']
 dataset_hash=hashlib.sha256()
 for p in files:
  label=ds/'labels/test'/(p.stem+'.txt');dataset_hash.update((p.name+sha(p)+sha(label)).encode())
 model=YOLO(str(weights));device='mps' if torch.backends.mps.is_available() else 'cpu'
 options={}
 if color=='bgr':
  from bgr_segmentation import BGRSegmentationValidator
  options['validator']=BGRSegmentationValidator
 metrics=model.val(**options,data=str(data),split='test',imgsz=640,conf=.001,iou=.7,batch=2,device=device,workers=0,augment=False,plots=True,project=str(out),name='standard_metrics',exist_ok=True,verbose=False)
 # Dataset verification may repair JPEG end markers before inference.
 dataset_hash=hashlib.sha256()
 for p in files:
  label=ds/'labels/test'/(p.stem+'.txt');dataset_hash.update((p.name+sha(p)+sha(label)).encode())
 perclass=[]
 for i,c in enumerate(metrics.box.ap_class_index):
  perclass.append({'class':names[int(c)],'box_precision':float(metrics.box.p[i]),'box_recall':float(metrics.box.r[i]),'box_mAP50':float(metrics.box.ap50[i]),'box_mAP50_95':float(metrics.box.ap[i]),'mask_precision':float(metrics.seg.p[i]),'mask_recall':float(metrics.seg.r[i]),'mask_mAP50':float(metrics.seg.ap50[i]),'mask_mAP50_95':float(metrics.seg.ap[i])})
 # Diagnostic image-label score: highest prediction, no confidence rejection.
 # AP above is a separate standard detection metric, not forced classification.
 rows=[];matrix=np.zeros((4,5),dtype=int)
 for p in files:
  source=str(p)
  if color=='bgr':
   import cv2
   # NumPy input is assumed BGR by the predictor and reversed into a tensor.
   # Supplying RGB here produces the required BGR tensor, exactly once.
   im=cv2.imread(str(p));assert im is not None
   source=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
  pred=model.predict(source=source,imgsz=640,conf=0.0,iou=.7,max_det=1,device=device,augment=False,rect=False,verbose=False)[0]
  truth=sorted({int(s.split()[0]) for s in (ds/'labels/test'/(p.stem+'.txt')).read_text().splitlines() if s.strip()})
  guess=int(pred.boxes.cls[0]) if len(pred.boxes) else None
  row={'image':p.name,'ground_truth_classes':[names[c] for c in truth],'prediction':names[guess] if guess is not None else None,'confidence':float(pred.boxes.conf[0]) if guess is not None else None,'single_class_ground_truth':len(truth)==1,'correct':guess==truth[0] if len(truth)==1 else None}
  rows.append(row)
  if len(truth)==1:matrix[truth[0],guess if guess is not None else 4]+=1
 single=[r for r in rows if r['single_class_ground_truth']];correct=sum(r['correct'] for r in single)
 assert sha(weights)==frozen
 result={'checkpoint':str(weights),'checkpoint_sha256':frozen,'test_dataset_sha256':dataset_hash.hexdigest(),'test_images':len(files),'standard_metric_settings':{'split':'test','imgsz':640,'confidence':.001,'iou':.7,'augment':False},'standard_metrics':{k:float(v) for k,v in metrics.results_dict.items()},'per_class':perclass,'highest_confidence_image_label':{'correct':correct,'evaluated':len(single),'accuracy':correct/len(single) if single else None,'excluded_empty_or_mixed':len(rows)-len(single),'matrix_rows':[names[i] for i in range(4)],'matrix_columns':[names[i] for i in range(4)]+['no_prediction'],'confusion_matrix':matrix.tolist(),'predictions':rows},'limitations':['Capture-session/near-duplicate leakage has not been audited.','Source labels are not all verified against the new project Agtron intervals.','Image-label agreement does not establish whole-pile or per-bean performance.','Forced classification uses top postprocessed PyTorch prediction; CPU ONNX parity/latency is not tested.'],'deployed':False}
 result['color_order']=color
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Held-out test evaluation',f'\nCheckpoint: {weights}',f'\nTest images: {len(files)}',f'\nHighest-confidence single-class image agreement: {correct}/{len(single)} ({correct/len(single):.2%}).', '\n## Standard detection / segmentation metrics']
 lines += [f'- {k}: {float(v):.4f}' for k,v in metrics.results_dict.items()]
 lines += ['\n## Limitations']+[f'- {s}' for s in result['limitations']]
 (out/'README.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'report':str(out),'correct':correct,'images':len(single),'metrics':result['standard_metrics']}),flush=True)
if __name__=='__main__':main()
