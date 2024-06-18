from PIL import Image
import cv2
import numpy as np
import cvzone
import torch.nn.functional as F

import sys
sys.path.append(r"./")
from utils.models import ModelManager

class Liveness(ModelManager):
    def __init__(self):
        super().__init__()


    def select_largest_face(self, face_boxes, whs):
        largest_area = 0
        largest_box = None
        for box, wh in zip(face_boxes, whs):
            area = wh[0] * wh[1]
            if area > largest_area:
                largest_area = area
                largest_box = box
        return largest_box

    def select_eyes_within_face(self, face_box, eye_boxes, cls_list):
        x1, y1, x2, y2 = face_box
        if not eye_boxes:
            return [], None
        for box, label in zip(eye_boxes, cls_list):
            if x1 <= box[0] <= x2 and y1 <= box[1] <= y2:
                return box, label
        return [], None

    def convert_cv2_to_pil(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)

    def save_cropped_face(self, face, filename):
        cv2.imwrite(filename, face)
        return face

    def compute_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        return iou

    def face_similarity(self, face1, face2):
        face1 = self.mtcnn(face1)
        face2 = self.mtcnn(face2)
        img_embedding1 = self.resnet(face1.unsqueeze(0))
        img_embedding2 = self.resnet(face2.unsqueeze(0))
        return F.cosine_similarity(img_embedding1, img_embedding2)
    
    def check_liveness(self, bbox1, bbox2, threshold1 ,face1, face2,threshold2 ):
        if self.compute_iou(bbox1, bbox2) > threshold1:
            # face1 = Image.open(face1_path)
            # face2 = Image.open(face2_path)
            face1 = self.convert_cv2_to_pil(face1)
            face2 = self.convert_cv2_to_pil(face2)
              
            similarity = self.face_similarity(face1, face2)
            if similarity > threshold2:
                return True
        return False

def face_detect(video_source, liveness: Liveness):
        
    print("Opening video...")
    cap = cv2.VideoCapture(video_source)
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
        results = liveness.yolo_model(frame, conf=0.4)
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
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
        
        if pass_liveness:
            print("All conditions satisfied.")
            break

if __name__ == "__main__":
    yolo_version = "yolov10"
    yolov10_weight_path = "weights/yolov10.pt"
    yolov8_weight_path = "weights/yolov8.pt"
    
    liveness = Liveness()
    liveness.load_yolo_model(yolo_version=yolo_version, yolov10_weight_path=yolov10_weight_path, yolov8_weight_path=yolov8_weight_path)
    liveness.load_recognition_model()
    
    face_detect(video_source=0, liveness=liveness)
