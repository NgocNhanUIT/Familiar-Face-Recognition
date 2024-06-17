import cv2
import cvzone
from ultralytics import YOLO

# video_source = r'C:\Users\NHAN\UIT_HK6\Nhan_dang\final_project\yolov9-face-detection\yolov9\runs\detect\exp2\0.mp4'
video_source = 0
face_detector = YOLO(r"..\weights\yolov8n-face.pt")

def face_detection(video_source):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True: 
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame = cv2.resize(frame, (840, 620))

        face_results = face_detector.predict(frame, conf=0.4)
        for infor in face_results:
            parameters = infor.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2-y1, x2-x1
                cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        
        
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        frame = cv2.resize(frame, (640, 480))

        face_results = face_detector.predict(frame, conf=0.4)
        for infor in face_results:
            parameters = infor.boxes
            for box in parameters:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2-y1, x2-x1
                cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
        
        out.write(frame)

    cap.release()
    out.release()
    print("Video processing completed.")

# face_detect(video_source)