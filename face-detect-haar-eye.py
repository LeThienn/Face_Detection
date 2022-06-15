import sys
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

blue = (247, 173, 62)
green = (120, 217, 30)

# Độ dày của hình chữ nhật được vẽ xung quanh các mặt
thickness = 2


def drawRectangle(image, color, faces, eyes):
    for (x, y, w, h) in faces:
        barLength = int(h / 8)
        barWidth = w
        cv2.rectangle(image, (x, y-barLength), (x+barWidth, y), color, -1)
        cv2.rectangle(image, (x, y-barLength),
                      (x+barWidth, y), color, thickness)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(image, (ex, ey), (ex+ew, ey+eh), green, thickness)
    return image


def detectFace(grayscale, image, isWebcam):
    #  Phát hiện khuôn mặt trực diện trong hình ảnh bằng cách sử dụng phân tầng khuôn mặt
    faces = face_cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
    )
    eyes = eye_cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
    )
    # Vẽ một hình chữ nhật xung quanh mặt chính diện được phát hiện
    image = drawRectangle(image, blue, faces, eyes)
    return image


def useWebcam():
    video = cv2.VideoCapture(0)
    while True:
        _, frame = video.read()
        # Convert frame to grayscale
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = detectFace(grayscale, frame, True)
        # Flip the frame
        frame = cv2.flip(frame, 1)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) > 0:
            break
    video.release()
    cv2.destroyAllWindows()

def useImage():
    # Read the image
    image = cv2.imread(sys.argv[1])
    # Convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = detectFace(grayscale, image, False)
    cv2.imshow("2.jpg", image)
    cv2.waitKey(0)

def main():
    if len(sys.argv) == 1:
        useWebcam()
    elif len(sys.argv) == 2:
        useImage()
    else:
        exit()

if __name__ == "__main__":
    main()
