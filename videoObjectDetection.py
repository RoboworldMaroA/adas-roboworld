# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import cv2

# Import YOLO
from ultralytics import YOLO

import numpy as np

import torch

print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())

cap = cv2.VideoCapture("dogs.mp4")
# can detect 80 different classes
model = YOLO("yolov8m.pt")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # find a position of the detected object

    #results = model(frame, device="mps")
    # results = model(frame, device="cpu")
    results = model(frame, device="mps")
    # results = model(frame)
    result = results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip(classes, bboxes):
        (x, y, x2, y2) = bbox

        cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 225), 2)
        if cls == 16:
            cls = "dog"
        if cls == 32:
            cls = "ball"
        cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    # esc to exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

'''

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
'''
