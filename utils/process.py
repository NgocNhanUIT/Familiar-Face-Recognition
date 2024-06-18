import cv2
import cvzone

import sys
sys.path.append(r"./")
from utils.liveness import Liveness

class Process(Liveness):
    def __init__(self):
        super().__init__()
        
    def process_webcam(self):
            
        print("Opening video...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        print("Video opened successfully.")
        prev_frame_face_box = None
        prev_eyes_closed = False
        prev_face = None
        pass_liveness = False
        
        while True: 
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            frame = cv2.resize(frame, (620, 500))
            results = self.yolo_model(frame, conf=0.4)
            face_boxes, whs, eye_boxes, cls_list_eyes, bboxs = [], [], [], [], []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w = y2 - y1, x2 - x1
                    bboxs.append((x1, y1, w, h))
                    cls = int(box.cls[0])
                    if cls == 1:
                        face_boxes.append((x1, y1, x2, y2))
                        whs.append((w, h))
                    elif cls == 0 or cls == 2:
                        cls_list_eyes.append(cls)
                        eye_boxes.append((x1, y1, x2, y2))

            if not face_boxes:
                continue

            largest_face_box = self.select_largest_face(face_boxes, whs)
            eyes_in_face, label = self.select_eyes_within_face(largest_face_box, eye_boxes, cls_list_eyes)
            cur_face = frame[largest_face_box[1]:largest_face_box[3], largest_face_box[0]:largest_face_box[2]]
            
            if len(eyes_in_face) > 0:
                if label == 0:
                    eyes_closed = True
                else:
                    eyes_closed = False

                if (prev_eyes_closed == True) and (eyes_closed == False):
                    self.save_cropped_face(prev_face, "./img/face_close.jpg")
                    self.save_cropped_face(cur_face, "./img/face_open.jpg")

                    pass_liveness = self.check_liveness(prev_frame_face_box, largest_face_box, 0.9, prev_face, cur_face, 0.95)
                prev_frame_face_box = largest_face_box
                prev_eyes_closed = eyes_closed
                prev_face = cur_face.copy()

            for box in bboxs:
                x, y, w, h = box
                cvzone.cornerRect(frame, [x, y, w, h], l=9, rt=3)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield frame
            
            if pass_liveness:
                yield 'liveness_passed'
                break

            # for box in bboxs:
            #     x, y, w, h = box
            #     cvzone.cornerRect(frame, [x, y, w, h], l=9, rt=3)
            #     cv2.imshow("frame", frame)
            #     cv2.waitKey(1)
        
            # if pass_liveness:
            #     print("All conditions satisfied.")
            #     break

            
    def process_video(input_path, output_path, liveness: Liveness):
        yolo = liveness.yolo_model
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        print("Video opened successfully.")
        prev_frame_face_box = None
        prev_eyes_closed = False
        prev_face = None
        pass_liveness = False
        
        while True: 
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            frame = cv2.resize(frame, (840, 620))
            results = yolo(frame, conf=0.5)
            face_boxes, whs, eye_boxes, cls_list_eyes, bboxs = [], [], [], [], []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w = y2 - y1, x2 - x1
                    bboxs.append((x1, y1, w, h))
                    cls = int(box.cls[0])
                    if cls == 1:
                        face_boxes.append((x1, y1, x2, y2))
                        whs.append((w, h))
                    elif cls == 0 or cls == 2:
                        cls_list_eyes.append(cls)
                        eye_boxes.append((x1, y1, x2, y2))

            if not face_boxes:
                continue

            largest_face_box = liveness.select_largest_face(face_boxes, whs)
            eyes_in_face, label = liveness.select_eyes_within_face(largest_face_box, eye_boxes, cls_list_eyes)
            cur_face = frame[largest_face_box[1]:largest_face_box[3], largest_face_box[0]:largest_face_box[2]]
            
            if len(eyes_in_face) > 0:
                if label == 0:
                    eyes_closed = True
                else:
                    eyes_closed = False

                if (prev_eyes_closed == True) and (eyes_closed == False):
                    liveness.save_cropped_face(prev_face, "./img/face_close.jpg")
                    liveness.save_cropped_face(cur_face, "./img/face_open.jpg")

                    pass_liveness = liveness.check_liveness(prev_frame_face_box, largest_face_box, 0.9, prev_face, cur_face, 0.95)
                prev_frame_face_box = largest_face_box
                prev_eyes_closed = eyes_closed
                prev_face = cur_face.copy()

            for box in bboxs:
                x, y, w, h = box
                cvzone.cornerRect(frame, [x, y, w, h], l=9, rt=3)
                
            if pass_liveness:
                print("All conditions satisfied.")
                break
            

        cap.release()
        out.release()
        print("Video processing completed.")


if __name__ == "__main__":
    
    yolo_version = "yolov8"
    yolov10_weight_path = "weights/yolov10.pt"
    yolov8_weight_path = "weights/yolov8.pt"
    
    process = Process()
    process.load_yolo_model(yolo_version=yolo_version, yolov10_weight_path=yolov10_weight_path, yolov8_weight_path=yolov8_weight_path)
    process.load_recognition_model()
    process.process_webcam()
    # process.process_video(input_path='./data/video.mp4', output_path='./data/result.mp4')