from ultralytics import YOLO
import plotly.express as px
import cv2
import easyocr
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

#loading the trained model
model = YOLO('/content/drive/MyDrive/best.pt')
#loading the tracker
tracker = DeepSort(max_age = 30)
#loading OCR
reader = easyocr.Reader(['en'])


#OOP Class for reusability
class vid:
  def __init__(self,path):
    self.path = path

  def frame_gen(self ):
    cap = cv2.VideoCapture(self.path)
    while cap.isOpened():
      ret,frame = cap.read()
      if ret:
        yield frame
      else:
        break

  def vid_gen(self):
    cap = cv2.VideoCapture(self.path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = cv2.VideoWriter(f'numplatestest.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=fps,
                          frameSize=(w,h))
    print(f'Vid gen built with total frames:{tf}')
    return out

def rgb(frame):
  return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  
#loading the sample video for inference
start = vid('car.mp4')
frames = start.frame_gen()
writer = start.vid_gen()


#Inference on the sample video
counted = set()
plates = {}
for frame in frames:
  res = model.predict(frame,conf=0.50,iou=0.70)[0]
  dts =[]
  for r in res.boxes:
    x1,y1,x2,y2 = r.xyxy[0]
    bbox = int(x1),int(y1),int(x2-x1),int(y2-y1)
    conf = r.conf[0]
    cls = r.cls[0]
    dts.append([bbox,conf,cls])
  tracks = tracker.update_tracks(dts,frame=frame)
  for track in tracks:
    l,t,r,b = track.to_ltrb()
    l,t,r,b = int(l),int(t),int(r),int(b)
    cid = track.det_class
    tid = track.track_id
    if tid not in counted:
      counted.add(tid)
      plates[tid] = frame[t:b,l:r]
    conf = track.det_conf
    if conf is not None:
      conf = round(conf,2)
    else:
      conf = 0.0
    cv2.rectangle(frame,(l,t),(r,b),color=(0,255,0),
                  thickness=2)
    label = f'#{tid}'
    cv2.putText(frame,label,(l,t-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,
                color=(0,255,0),thickness=3)
  writer.write(frame)
writer.release()