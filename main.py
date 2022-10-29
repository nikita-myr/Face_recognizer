import cv2
import datetime as dt
import numpy as np
import os

CASCADE_PATH = 'filters/haarcascade_frontalface_default.xml'
CLF = cv2.CascadeClassifier(CASCADE_PATH)


class Camera:
    def __init__(self, capture, width, heght, fps):
        self.capture = capture
        self.width = width
        self.height = heght
        self.fps = fps

    capture = cv2.VideoCapture(0)

    width = int(capture.get(3))
    height = int(capture.get(4))
    fps = int(capture.get(5))

    def frames_out():
        ret, frame = Camera.capture.read()
        return frame


class Recording:
    def __init__(self, count, output_video, frame):
        self.count = count
        self.output_video = output_video
        self.frame = frame

    file_name = f'{dt.datetime.now()}.avi'
    file_path = os.path.abspath('main.py')[:-7] + '/video'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(f'{file_path}/{file_name}',
                                   fourcc, Camera.fps,
                                   (Camera.width, Camera.height))


if __name__ == '__main__':
    while True:
        input_signal = Camera.frames_out()

        gray = cv2.cvtColor(input_signal, cv2.COLOR_BGR2GRAY)

        faces = CLF.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=10,
                                     minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE
                                     )

        for (x, y, width, height) in faces:
            cv2.rectangle(input_signal, (x, y), (x + width, y + height),
                          (255, 255, 0), 2)

        if np.all(faces) == True:
            Recording.output_video.write(input_signal)

        cv2.imshow('face_recognize', input_signal)

        if cv2.waitKey(1) == ord('q'):
            Camera.capture.release()
            cv2.destroyAllWindows()
            break
