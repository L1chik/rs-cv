import cv2
import face_detection as rs
import numpy as np

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# test = cv2.imread('DSFD_demo2.jpg')

while True:
    success, img = vid.read()

    # Convert img to bytes and send to rust processing
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
    bytes_rs = rs.rs_faces(img_bytes)

    # Convert bytes to image
    byte_array = bytearray(bytes_rs)
    img_rs = np.array(byte_array).reshape(480, 640, 3)
    img_rs = cv2.cvtColor(img_rs, cv2.COLOR_RGB2BGR)

    cv2.imshow('img', img_rs)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()