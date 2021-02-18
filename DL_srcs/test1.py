import cv2, dlib, imutils
import numpy as np
from imutils import face_utils
# from keras.models import load_model
import tensorflow as tf
import time

IMG_SIZE = (34, 26)     # 입력 이미지 크기

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = tf.keras.models.load_model('2018_12_17_22_58_35.h5')                # 모델 로드
model.summary()


def crop_eye(img, eye_points):
    x1, y1 = np.amin(eye_points, axis=0)
    x2, y2 = np.amax(eye_points, axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = (x2 - x1) * 1.2
    h = w * IMG_SIZE[1] / IMG_SIZE[0]

    margin_x, margin_y = w / 2, h / 2

    min_x, min_y = int(cx - margin_x), int(cy - margin_y)
    max_x, max_y = int(cx + margin_x), int(cy + margin_y)

    eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

    eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

    return eye_img, eye_rect


# main
cap = cv2.VideoCapture(0)   # 웹캠 영상 연결

check = False
start = 0
while cap.isOpened():
    ret, img_ori = cap.read()

    if not ret:
        break

    img_ori = cv2.resize(img_ori, dsize=(500, 500), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        cv2.imshow('l', eye_img_l)
        cv2.imshow('r', eye_img_r)

        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)

        if pred_l < 0.2 and pred_r < 0.2:
            if check == False:
                start = time.time()
                check = True
            if (time.time() - start > 3):  # 시작좌표, 폰트종류, 폰트크기, 색상, 사이즈, 선 유형
                time_temp = time.time()
                while time.time() - time_temp < 3:
                    cv2.putText(img, 'sleep state!!!!!', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                start = 0
                check = False
        else:
            start = 0
            check = False

        # visualize
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255, 255, 255), thickness=2)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255, 255, 255), thickness=2)

        cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('result', img)
    if cv2.waitKey(1) == 48:  # 0입력시 종료
        break
cap.release()
cv2.destroyAllWindows()
