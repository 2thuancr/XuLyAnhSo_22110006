import cv2
import numpy as np
import imutils
import streamlit as st



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
st.markdown(page_bg_img,unsafe_allow_html=True)
st.title("Nhận dạng màu sắc")
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
warmup_frames = 60
FRAME_WINDOW = st.image([])
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    lower_Yellow = np.array([25,70,125])
    upper_Yellow = np.array([35,255,255])

    lower_Green = np.array([65, 60, 60])
    upper_Green = np.array([80, 255, 255])

    lower_Red = np.array([0,70,150])
    upper_Red = np.array([15,255,255])

    lower_Blue = np.array([90,60,70])
    upper_Blue = np.array([114,255,255])

    lower_Purple = np.array([[125, 30, 50]])
    upper_Purple = np.array([140, 255, 255])

    lower_Pink = np.array([150, 70, 70])
    upper_Pink = np.array([180, 255, 255])

    lower_White = np.array([0, 0, 200])
    upper_White = np.array([180, 50, 255])

    lower_Black = np.array([0, 0, 0])
    upper_Black = np.array([180, 255, 30])

    lower_Grey = np.array([0, 0, 40])
    upper_Grey = np.array([180, 30, 200])


    mask1 = cv2.inRange(hsv, lower_Yellow, upper_Yellow)
    mask2 = cv2.inRange(hsv, lower_Green, upper_Green)
    mask3 = cv2.inRange(hsv, lower_Red, upper_Red)
    mask4 = cv2.inRange(hsv, lower_Blue, upper_Blue)
    mask5 = cv2.inRange(hsv, lower_Purple, upper_Purple)
    mask6 = cv2.inRange(hsv, lower_Pink, upper_Pink)
    mask7 = cv2.inRange(hsv, lower_White, upper_White)
    mask8 = cv2.inRange(hsv, lower_Black, upper_Black)
    mask9 = cv2.inRange(hsv, lower_Grey, upper_Grey)

    cnts1 = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    cnts2 = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    cnts3 = cv2.findContours(mask3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)

    cnts4 = cv2.findContours(mask4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)

    cnts5 = cv2.findContours(mask5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts5 = imutils.grab_contours(cnts5)

    cnts6 = cv2.findContours(mask6, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts6 = imutils.grab_contours(cnts6)

    cnts7 = cv2.findContours(mask7, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts7 = imutils.grab_contours(cnts7)

    cnts8 = cv2.findContours(mask8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts8 = imutils.grab_contours(cnts8)

    cnts9 = cv2.findContours(mask9, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts9 = imutils.grab_contours(cnts9)

    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 > 5000:

            cv2.drawContours(frame, [c], -1, (0,255,0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            cv2.circle(frame, (cx,cy), 7, (255,255,255), -1)
            cv2.putText(frame, "Yellow", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 3)

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "green", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts3:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "red", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts4:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "blue", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts5:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "purple", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    for c in cnts6:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "pink", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    for c in cnts7:
        area2 = cv2.contourArea(c)
        if area2 > 5000:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "white", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    for c in cnts8:
        area2 = cv2.contourArea(c)
    if area2 > 5000:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "black", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)
    for c in cnts9:
        area2 = cv2.contourArea(c)
    if area2 > 5000:
        cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

        M = cv2.moments(c)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
        cv2.putText(frame, "grey", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    FRAME_WINDOW.image(frame, channels='BGR')


cap.release()
cv2.destroyAllWindows()
