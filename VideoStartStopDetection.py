import os

from ultralytics import YOLO
import cv2

import torch

print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())


#This app read the video and recognize object like:
#  0: 'start', 1: 'go', 2: 'stop', 3: 'turnback', 4: 'turnLeft', 5: 'turnRight', 6: 'turnLeft2', 7: 'turnRight2', 8: 'turnback2', 9: 'start2', 10: 'stop2'}
# When recognize then save on the copy of the video predicted annotation with the name of the object was predicted
#model train27 is a model created by yolo using main.py program
VIDEOS_DIR = os.path.join('.', '')

video_path = os.path.join(VIDEOS_DIR, 'startStop.mp4')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
# Had error try to change something in this line
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
model_path = os.path.join('.', 'runs', 'detect', 'train27', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

class_name_dict = {0: 'start', 1: 'go', 2: 'stop', 3: 'turnback', 4: 'turnLeft', 5: 'turnRight', 6: 'turnLeft2', 7: 'turnRight2', 8: 'turnback2', 9: 'start2', 10: 'stop2'}

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()
    
    # Display Video If you display then video is slow and do not see predictions
    # cv2.imshow("Img", frame)
    # key = cv2.waitKey(1)
    # # esc to exit
    # if key == 27:
    #     break


cap.release()
out.release()
cv2.destroyAllWindows()