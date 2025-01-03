#In terminal ty command: 
# cd yolov8-web-app-youtube 
# export FLASK_APP=CheckEnvironment.py

from flask import Flask
# import os

app = Flask(__name__)
application = app

@app.route('/')
def hello_world():
    return 'Hello world!'


# if __name__ == '__main__':
#     os.environ.setdefault('FLASK_ENV', 'development')
#     app.run(debug=False, port=5001, host='0.0.0.0')

if __name__ == '__main__':
    app.run(debug=True)