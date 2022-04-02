import cv2
import face_detection as rs
import numpy as np

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    success, img = vid.read()

    # Convert img to bytes and send to rust processing
    img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
    bytes_rs = rs.to_gray(img_bytes)

    # Convert bytes to image
    byte_array = bytearray(bytes_rs)
    img_rs = np.array(byte_array).reshape(480, 640)
    img_rs = cv2.cvtColor(img_rs, cv2.COLOR_GRAY2BGR)

    # Test: return np array from rust - delay ~0.5 sec
    # nparr = np.array(bytes_rs).reshape(480, 640, 4)

    # print(bytes_rs)
    # print(img.shape)

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()