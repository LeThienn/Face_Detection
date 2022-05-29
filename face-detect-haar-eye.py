import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

blue = (247, 173, 62)
green = (120, 217, 30)

# Độ dày của hình chữ nhật được vẽ xung quanh các mặt
thickness = 2

def drawRectangle(image, color, faces):
    for (x, y, w, h) in faces:
        barLength = int(h / 8)
        barWidth = w
        cv2.rectangle(image, (x, y-barLength), (x+barWidth, y), color, -1)
        cv2.rectangle(image, (x, y-barLength),
                      (x+barWidth, y), color, thickness)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    return image


def detectFace(grayscale, image, isWebcam):
    #  Phát hiện khuôn mặt trực diện trong hình ảnh bằng cách sử dụng phân tầng khuôn mặt
    faces = face_cascade.detectMultiScale(
        grayscale,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if not(isWebcam):
        # Phát hiện khuôn mặt hồ sơ trong hình ảnh bằng cách sử dụng phân tầng khuôn mặt
        profileFaces = eye_cascade.detectMultiScale(
            grayscale,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30),
        )
        # Phát hiện khuôn mặt hồ sơ trong hình ảnh được lật để phát hiện khuôn mặt hồ sơ hướng sang phải
        flipped = cv2.flip(grayscale, 1)
        profileFacesFlipped = eye_cascade.detectMultiScale(
            flipped,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30)
        )


# cap = cv2.VideoCapture(0)

# while 1:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]

#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex, ey, ew, eh) in eyes:
#             cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

#     cv2.imshow('img', img)
#     ch = cv2.waitKey(1)
#     if ch & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
