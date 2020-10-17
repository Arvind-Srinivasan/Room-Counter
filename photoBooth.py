import cv2

cv2.namedWindow("preview")

vc = cv2.VideoCapture(0)

while True:
    rval, frame = vc.read()
    cv2.imshow("preview", frame)
    if (cv2.waitKey(1) == 27):
        break

cv2.destroyWindow("preview")
vc.release()
