from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
from liveness import face_detect
from faceDetect import process_video
from utils.manage_db import FaceDatabaseManager

from werkzeug.utils import secure_filename
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = r'.\static\upload'
RESULT_FOLDER = r'.\static\result'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
video_path = None
result_path = None
file_path = r"./data/face_db.csv"
manager = FaceDatabaseManager(face_db_path=file_path)
face_path = r"./img/face_open.jpg"

# Global variable to track liveness status
liveness_passed = False

def recognize(img_path):
    img_embedding, img_cropped = manager.get_embedding(img_path)
    cropped_img_path = os.path.join(app.config['RESULT_FOLDER'], "face_cropped.jpg")
    manager.save_img(img_cropped, cropped_img_path)
    find_indices, find_distances, names = manager.find_k_nearest_neighbors(img_embedding, k=3, threshold=0.3)
    if len(find_indices) > 0:
        return names[0], cropped_img_path
    else:
        return "Unknown", cropped_img_path

@app.route('/')
def index():
    global liveness_passed
    liveness_passed = False  # Reset liveness check on load
    return render_template('webcam.html', yolo_version='v8', pass_liveness=False)

@app.route('/video')
def index_video():
    return render_template('video.html', video_filename='')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, result_path
    if 'videoFile' not in request.files:
        return 'No file part'
    file = request.files['videoFile']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        file.save(video_path)
        process_video(video_path, result_path)  # Process the video to detect faces and save the result
        return redirect(url_for('show_video', video_filename=filename))

@app.route('/show_video/<video_filename>', methods=['GET'])
def show_video(video_filename):
    return render_template('video.html', video_filename=video_filename)
    
@app.route('/video_feed')
def video_feed():
    yolo_ver = request.args.get('yolo-version')
    pass_liveness = request.args.get('pass-liveness', 'false').lower() == 'true'

    def generate_frames():
        global liveness_passed
        for frame in face_detect(0, yolo_ver, pass_liveness=pass_liveness):
            if frame == 'liveness_passed':
                liveness_passed = True
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/yolo_version', methods=['POST'])
def yolo_version():
    yolo_ver = request.form.get('yolo-version')
    return render_template('webcam.html', yolo_version=yolo_ver)

@app.route('/get_pass_liveness', methods=['GET'])
def get_pass_liveness():
    global liveness_passed
    return jsonify(pass_liveness=liveness_passed)

@app.route('/set_pass_liveness', methods=['POST'])
def set_pass_liveness():
    global liveness_passed
    liveness_passed = False
    return jsonify(pass_liveness=liveness_passed)

@app.route('/recognize', methods=['POST'])
def recognize_route():
    img_path = request.form['img_path']
    name, cropped_img_path = recognize(img_path)
    return jsonify({'name': name, 'img_path': cropped_img_path})

if __name__ == '__main__':
    app.run(debug=True)
