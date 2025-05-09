import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import streamlit as st
import random


page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-040.jpg");
    background-position: center;
}
</style>
""" 
st.set_page_config(page_title="Nhận dạng cử chỉ")

st.markdown(page_bg_img,unsafe_allow_html=True)
st.title("Nhận biết cử chỉ")
label = "Warm up...."
n_time_steps = 10
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = None
try:
    model = tf.keras.models.load_model("D:/XulyAnh/WebDemo/NhanDangCuChi/model.h5")
except Exception as e:
    print("Error loading the model:", e)

cap = cv2.VideoCapture(0)


# Tạo dấu thời gian
def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    if results[0][0] > 0.5:
        label = "SWING BODY"
    else:
        label = "SWING HAND"
    return label


i = 0
warmup_frames = 60
FRAME_WINDOW = st.image([])
mylist = ["SWING BODY", "SWING HAND"]
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # Check if the model is loaded
                if model is not None:
                    # predict
                    t1 = threading.Thread(target=detect, args=(model, lm_list))
                    t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)
    img = draw_class_on_image(label, img)
    FRAME_WINDOW.image(img, channels='BGR')

cap.release()
cv2.destroyAllWindows()