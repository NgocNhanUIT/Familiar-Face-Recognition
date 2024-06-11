from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from faceDetect import face_detect
from faceDetect import process_video
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

UPLOAD_FOLDER = r'C:\Users\NHAN\UIT_HK6\Nhan_dang\final_project\static\upload'
RESULT_FOLDER = r'C:\Users\NHAN\UIT_HK6\Nhan_dang\final_project\static\result'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
video_path = None
result_path = None

@app.route('/')
def index():
    return render_template('webcam.html')

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
    # result_video_path = os.path.join(app.config['RESULT_FOLDER'], video_filename)
    # if not os.path.exists(result_video_path):
    #     return "Result video not found", 404
    return render_template('video.html', video_filename=video_filename)
    
@app.route('/video_feed')
def video_feed():
    return Response(face_detect(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
