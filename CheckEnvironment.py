#In terminal ty command: 
# cd yolov8-web-app-youtube 
# export FLASK_APP=CheckEnvironment.py

from flask import Flask
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello world!'