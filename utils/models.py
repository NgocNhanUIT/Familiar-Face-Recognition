from ultralytics import YOLOv10, YOLO

import sys
sys.path.append(r"./")
from facenet_pytorch import MTCNN, InceptionResnetV1

class ModelManager:
    def __init__(self):
        self.yolo_model, self.mtcnn, self.resnet = None, None, None

    def load_yolo_model(self, yolo_version, yolov10_weight_path, yolov8_weight_path):
        print("Loading models...")
        if yolo_version == "yolov10":
            self.yolo_model = YOLOv10(yolov10_weight_path)
        elif yolo_version == "yolov8":
            self.yolo_model = YOLO(yolov8_weight_path)
        print(f"Models {yolo_version} loaded successfully.")
        
    def load_recognition_model(self, mtcnn_image_size=160, mtcnn_margin=0, pretrained='vggface2'):
        self.mtcnn = MTCNN(image_size=mtcnn_image_size, margin=mtcnn_margin)
        self.resnet = InceptionResnetV1(pretrained=pretrained).eval()