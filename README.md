# Familiar-Face-Recognition

## Create a New Environment

To create and activate a new Conda environment for the project, use the following commands:

```bash
conda create --name face-reg python=3.8
conda activate face-reg
```

## Install Requirements
```bash
pip install -r requirements.txt
git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
pip install git+https://github.com/THU-MIG/yolov10.git
```

## Download YOLOv8 Pretrained Weights
1. Visit the https://github.com/akanametov/yolo-face/tree/dev
2. Download yolov8n-face.pt model
```plaintext
Familiar-Face-Recognition/
├── static
│   ├── css/
│   ├── scripts/
│   ├── upload/
│   └── result/
├── templates/
│   ├── video.html
│   ├── webcam.html
│   └── ...
├── requirements.txt
├── .gitignore
├── app.py
├── faceDetect.py
├── README.md
└── yolov8n-face.pt
```

## Inference
```bash
python app.py
```
or
```bash
flask run
```
