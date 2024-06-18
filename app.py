from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import cv2
from utils.manage_db import ManageDB
from utils.process import Process
import sqlite3
from datetime import datetime
import cvzone
from werkzeug.utils import secure_filename
import os
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = r'.\static\upload'
RESULT_FOLDER = r'.\static\result'
IMAGE_FOLDER = r'.\static\images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER
video_path = None
result_path = None
file_path = r"./data/face_db.csv"
face_path = r"./img/face_open.jpg"
yolov10_weight_path = "weights/yolov10.pt"
yolov8_weight_path = "weights/yolov8.pt"
process = Process()
process.load_recognition_model()
manager = ManageDB(face_db_path=file_path)
manager.load_recognition_model()

# Global variable to track liveness status
liveness_passed = False 
face_frame = None

def recognize(img_path):
    img_embedding, img_cropped = manager.get_embedding(img_path)
    cropped_img_path = os.path.join(app.config['RESULT_FOLDER'], "face_cropped.jpg")
    manager.save_img(img_cropped, cropped_img_path)
    find_indices, find_distances, names = manager.find_k_nearest_neighbors(img_embedding, k=3, threshold=0.8)
    if len(find_indices) > 0:
        manager.attendance(names[0])
        return names[0], cropped_img_path, find_distances[0]
    else:
        return "Unknown", cropped_img_path, 0

@app.route('/')
def index():
    global liveness_passed
    liveness_passed = False  # Reset liveness check on load
    return render_template('webcam.html', yolo_version='v8')

@app.route('/video')
def index_video():
    return render_template('video.html', video_filename='')

@app.route('/register')
def index_register():
    return render_template('register.html')

@app.route('/tracker')
def index_tracker():
    return render_template('attendance.html', selected_date='', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')

    conn = sqlite3.connect('./data/attendance.db')
    cursor = conn.cursor()

    cursor.execute("SELECT name, time FROM attendance WHERE date = ?", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('attendance.html', selected_date=selected_date, no_data=True)
    
    return render_template('attendance.html', selected_date=selected_date, attendance_data=attendance_data)

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
    # pass_liveness = request.args.get('pass-liveness', 'false').lower() == 'true'
    if yolo_ver == 'v8':
        yoloversion = "yolov8"
    else:
        yoloversion = "yolov10"
    process.load_yolo_model(yolo_version=yoloversion, yolov10_weight_path=yolov10_weight_path, yolov8_weight_path=yolov8_weight_path)
    def generate_frames():
        global liveness_passed
        for frame in process.process_webcam():
            if frame == 'liveness_passed':
                liveness_passed = True
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    global cap, face_frame
    cap = cv2.VideoCapture(0)
    process.load_yolo_model(yolo_version='yolov8', yolov10_weight_path=yolov10_weight_path, yolov8_weight_path=yolov8_weight_path)
    def generate_frame():
        global face_frame
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                frame = cv2.resize(frame, (620, 500))
                face_frame = cv2.imencode('.jpg', frame)[1].tobytes()
                results = process.yolo_model(frame, conf=0.4)
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls == 1:
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            h, w = y2 - y1, x2 - x1
                            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
    name, cropped_img_path, distance = recognize(img_path)
    return jsonify({'name': name, 'img_path': cropped_img_path, 'distance': float(distance)})

@app.route('/get_face', methods=['POST'])
def get_face():
    global cap, face_frame
    if face_frame is not None:
        # Save the current frame
        nparr = np.frombuffer(face_frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect face in the frame
        results = process.yolo_model(img, conf=0.4)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls == 1:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cropped_face = img[y1:y2, x1:x2]
                    face_img_path = os.path.join(IMAGE_FOLDER, "register_image.jpg")
                    cv2.imwrite(face_img_path, cropped_face)
                    cap.release()  # Stop the webcam
                    return jsonify(success=True, img_path=face_img_path)
    return jsonify(success=False)

@app.route('/register_face', methods=['POST'])
def register_face():
    name = request.form['name']
    img_path = os.path.join(IMAGE_FOLDER, "register_image.jpg")
    manager.add_face_db(img_path, name)
    return jsonify(success=True)

if __name__ == '__main__':
    app.run(debug=True)
