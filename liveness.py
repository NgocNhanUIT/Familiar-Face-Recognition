import cv2
import torch
from ultralytics import YOLOv10
import cvzone

model = YOLOv10(r'./best.pt')

def face_detect(video_source):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True: 
        ret, video = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        video = cv2.resize(video, (840, 620)) # nay sua kich thuuoc cuua so, e de hien thij tren web ne hoi nho

        face_results = model(video, conf=0.4)
        for infor in face_results:
            cls = infor.boxes.cls
            print(cls)
            parameters = infor.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2-y1, x2-x1
                cvzone.cornerRect(video, [x1, y1, w, h], l=9, rt=3)

                
        # ret, buffer = cv2.imencode('.jpg', video)
        # frame = buffer.tobytes()
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
        if 0 in cls:
            print("Face detected")
            break
        
        cv2.imshow("Video", video)  
        cv2.waitKey(1)
        
    cap.release()
    cv2.destroyAllWindows()
        
        
face_detect(0)