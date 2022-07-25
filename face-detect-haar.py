# Usage: python face-detect-haar.py [optional.jpg]

import cv2
import sys

# Load the cascade classifiers
# Tải các bộ phân loại
frontalFaceCascade = cv2.CascadeClassifier(
    "haarcascades/haarcascade_frontalface_default.xml")
profileFaceCascade = cv2.CascadeClassifier(
    "haarcascades/haarcascade_profileface.xml")

# màu BGR
blue = (247, 173, 62)
green = (120, 217, 30)

# Độ dày của hình chữ nhật được vẽ xung quanh các mặt
thickness = 2

def drawRectangle(image, color, faces):
    for (x, y, w, h) in faces:
        barLength = int(h / 8)
        barWidth = w
        # mảng xanh trên đầu của hình chữ nhẬT
        cv2.rectangle(image, (x, y-barLength), (x+barWidth, y), color, -1) 
        cv2.rectangle(image, (x, y-barLength),
                      (x+barWidth, y), color, thickness)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    return image


def detectFace(grayscale, image, isWebcam):
    #  Phát hiện khuôn mặt trực diện trong hình ảnh bằng cách sử dụng phân tầng khuôn mặt
    faces = frontalFaceCascade.detectMultiScale(
        grayscale,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(30, 30),
    )
    if not(isWebcam):
        # Phát hiện khuôn mặt hồ sơ trong hình ảnh bằng cách sử dụng phân tầng khuôn mặt
        profileFaces = profileFaceCascade.detectMultiScale(
            grayscale,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30),
        )
        # Phát hiện khuôn mặt hồ sơ trong hình ảnh được lật để phát hiện khuôn mặt hồ sơ hướng sang phải
        flipped = cv2.flip(grayscale, 1)
        profileFacesFlipped = profileFaceCascade.detectMultiScale(
            flipped,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(30, 30)
        )
    # Vẽ một hình chữ nhật xung quanh mặt chính diện được phát hiện
    imageDraw = drawRectangle(image, blue, faces)
    if not (isWebcam):
        # Vẽ một hình chữ nhật xung quanh mỗi mặt được phát hiện
        
        image = drawRectangle(imageDraw, blue, profileFaces)
        image = cv2.flip(imageDraw, 1)
        image = drawRectangle(imageDraw, blue, profileFacesFlipped)
        image = cv2.flip(image, 1)
        
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
