from flask import Flask, render_template, request
import numpy as np
import base64
import cv2
import os

from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# model = YOLO('yolo11n.pt')
# development model url
model = YOLO('adasRoboworld/start_stop_yolo8.pt')
# Deploy model URL
# model = YOLO('start_stop_yolo8.pt')
# model = YOLO('runs/detect/train27/weights/last.pt')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_VIDEO_EXTENSIONS = {'MP4', 'mp4', 'avi'}
app = Flask(__name__)
#this line is for production
application = app



#diffrent Version
from flask import send_file, Response
from werkzeug.utils import secure_filename
import io
from PIL import Image



app.config['UPLOAD_FOLDER'] = 'adasRoboworld/uploads'

class Detection:
    def __init__(self):
        #download weights from here:https://github.com/ultralytics/ultralytics and change the path
        self.model = YOLO(r"/Users/marek/Programowanie/Yolo8MacBookM1SiliconFlask/adasRoboworld/yolov8n.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)

        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img


detection = Detection()
@app.route('/imageV2')
def index():
    return render_template('imageV2.html')

@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part'

    file = request.files['image']
    if file.filename == '':
        return 'No selected file'

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (512, 512))
        img = detection.detect_from_image(img)
        output = Image.fromarray(img)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        os.remove(file_path)
        return send_file(buf, mimetype='image/png')


@app.route('/video')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (512, 512))
        if frame is None:
            break
        frame = detection.detect_from_image(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


#End Diffrent version
#############################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def predict_on_image(image_stream):
    image = cv2.imdecode( np.asarray(bytearray(image_stream.read()), dtype=np.uint8) , cv2.IMREAD_COLOR)
    
    # //you can pick what class you looking for add classes=0 then it will recognize start
    # results = model.predict(image, classes=0, conf=0.3)
    #recognize all classes that confidence is more then 60%
    results = model.predict(image, conf=0.65)
    print(results[0])
    for image, result in enumerate(results):
        im_bgr = result.plot(conf=True)
        # print(result)
        # print(result.names)

    return im_bgr

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):

            predicted_image = predict_on_image(file.stream)

            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            file.stream.seek(0)
            original_img_base64 = base64.b64encode(file.stream.read()).decode('utf-8')

            return render_template('result.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)

    return render_template('index.html')

# Video stream route
@app.route('/videoStream', methods=['GET', 'POST'])
def homeVideoStream():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('videoStream.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('videoStream.html', error='No selected file')

        if file and allowed_video_file(file.filename):

            predicted_image = predict_on_image(file.stream)

            retval, buffer = cv2.imencode('.png', predicted_image)
            detection_img_base64 = base64.b64encode(buffer).decode('utf-8')

            file.stream.seek(0)
            original_img_base64 = base64.b64encode(file.stream.read()).decode('utf-8')

            return render_template('resultVideoStream.html', original_img_data=original_img_base64, detection_img_data=detection_img_base64)

    return render_template('videoStream.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=5001, host='0.0.0.0')


