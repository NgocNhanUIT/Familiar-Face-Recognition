# Familiar-Face-Recognition

## Create a New Environment

To create and activate a new Conda environment for the project, use the following commands:

```bash
conda create --name face-reg python=3.9
conda activate face-reg
```

## Install Requirements
```bash
pip install -r requirements.txt
git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
pip install git+https://github.com/THU-MIG/yolov10.git
```

## Download YOLOv8 Pretrained Weights
1. Download weight [yolov10](https://github.com/NgocNhanUIT/Familiar-Face-Recognition/releases/download/yolov10_weight/yolov10_weight.pt) or [yolov8](https://github.com/NgocNhanUIT/Familiar-Face-Recognition/releases/download/yolov10_weight/yolov8_weight.pt).
2. 
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
├── utils/
│   ├── liveness.py
│   ├── models.py
│   └── ...
├── requirements.txt
├── .gitignore
├── app.py
├── README.md
```

## Inference
```bash
python app.py
```
or
```bash
flask run
```
