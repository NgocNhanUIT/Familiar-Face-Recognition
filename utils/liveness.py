from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLOv10, YOLO
import cvzone
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

yolov10_weight_path = r"C:\Users\admin\Downloads\yolov10_weight.pt"
yolov8_weight_path = r"C:\Users\admin\Downloads\yolov8_weight.pt"

def select_largest_face(face_boxes,whs):
    largest_area = 0
    largest_box = None
    for box,wh in zip(face_boxes,whs):
        area = wh[0] * wh[1]
        if area > largest_area:
            largest_area = area
            largest_box = box
    return largest_box

def select_eyes_within_face(face_box, eye_boxes, cls_list):
    x1, y1, x2, y2 = face_box
    
    if eye_boxes == []:
        return [], None
    # eyes = [box for box in eye_boxes if x1 <= box[0] <= x2 and y1 <= box[1] <= y2]
    for box,label in zip(eye_boxes,cls_list):
        if x1 <= box[0] <= x2 and y1 <= box[1] <= y2:
            return box, label

def save_cropped_face(face, filename):
    cv2.imwrite(filename, face)
    return face

def compute_iou(box1, box2):
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

def face_similarity(face1, face2, mtcnn, resnet):
    face1 = mtcnn(face1)
    face2 = mtcnn(face2)

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding1 = resnet(face1.unsqueeze(0))
    img_embedding2 = resnet(face2.unsqueeze(0))
    return F.cosine_similarity(img_embedding1, img_embedding2)

def load_model(yolo_version):
    print("Loading models...")
    if yolo_version == "yolov10":
        model = YOLOv10(yolov10_weight_path)
    elif yolo_version == "yolov8":
        model = YOLO(yolov8_weight_path)
    # model = YOLOv10(yolo_weight_path)
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    print("Models loaded successfully.")
    return model, mtcnn, resnet



def face_detect(video_source):
    
    model,mtcnn,resnet = load_model(yolo_version="yolov8")
    
    print("Opening video...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    print("Video opened successfully.")
    # T = 0
    prev_frame_face_box = None
    # prev_frame_face = None
    prev_eyes_closed = False
    # prev_frame = None
    prev_face = None
    pass_liveness = False
    while True: 
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame = cv2.resize(frame, (840, 620))
        results = model(frame, conf=0.4)

        face_boxes = []
        cur_face = None
        whs = []
        eye_boxes = []
        cls_list_eyes = []
        bboxs = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                h, w = y2-y1, x2-x1
                bboxs.append((x1, y1, w, h))
                
                cls = int(box.cls[0])
                if cls == 1:
                    face_boxes.append((x1, y1, x2, y2))
                    whs.append((w, h))
                elif cls == 0 or cls == 2:
                    cls_list_eyes.append(cls)
                    eye_boxes.append((x1, y1, x2, y2))

        # print(cls_list_eyes)
        if not face_boxes:
            continue
        # if not eye_boxes:
        #     continue
        largest_face_box = select_largest_face(face_boxes, whs)
        eyes_in_face, label = select_eyes_within_face(largest_face_box, eye_boxes, cls_list_eyes)
        cur_face = frame[largest_face_box[1]:largest_face_box[3], largest_face_box[0]:largest_face_box[2]]
        # print(label)
        if len(eyes_in_face) > 0:
            if label == 0:
                eyes_closed = True
                save_cropped_face(cur_face, "./img/face_close.jpg")
            else:
                eyes_closed = False
            # print(prev_eyes_closed, eyes_closed )
            if (prev_eyes_closed == True) and (eyes_closed == False):
                
                save_cropped_face(cur_face, "./img/face_open.jpg")

                
                iou = compute_iou(prev_frame_face_box, largest_face_box)
                print(f"iou: {iou}")
                if iou > 0.9:
                    cur_face = Image.open("./img/face_open.jpg")
                    prev_face = Image.open("./img/face_close.jpg")
                    if prev_face is not None:
                        similarity = face_similarity(prev_face, cur_face, mtcnn, resnet)
                        print("Similarity: ", similarity)
                        if similarity > 0.95:
                            pass_liveness = True
                            # break

            prev_frame_face_box = largest_face_box
            prev_eyes_closed = eyes_closed
            prev_face = cur_face

        for box in bboxs:
            x, y, w, h = box
            cvzone.cornerRect(frame, [x, y, w, h], l=9, rt=3)
            
        # ret, buffer = cv2.imencode('.jpg', frame)
        # frame = buffer.tobytes()
        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        
        if pass_liveness:
            print("All conditions satisfied.")
            break
        
        
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
    
if __name__ == "__main__":
    face_detect(0)

