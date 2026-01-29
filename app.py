import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="FocusSense", layout="centered")
st.title("🎯 FocusSense - AI Study Focus Monitor")
st.write("MediaPipe based focus detection system")

# --------------------
# MediaPipe setup
# --------------------
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face = mp_face.FaceMesh(refine_landmarks=False)
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2)

# --------------------
# Utils
# --------------------
def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye):
    v1 = distance(landmarks[eye[1]], landmarks[eye[5]])
    v2 = distance(landmarks[eye[2]], landmarks[eye[4]])
    h = distance(landmarks[eye[0]], landmarks[eye[3]])
    return (v1 + v2) / (2.0 * h)

# --------------------
# Session log
# --------------------
if "logs" not in st.session_state:
    st.session_state.logs = []

# --------------------
# Camera Input
# --------------------
img_file = st.camera_input("📷 Turn on camera")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face.process(rgb)
    pose_res = pose.process(rgb)
    hand_res = hands.process(rgb)

    sleepy = distracted = bad_posture = stressed = False

    # -------- FACE --------
    if face_res.multi_face_landmarks:
        for fl in face_res.multi_face_landmarks:
            lm = fl.landmark
            ear = (eye_aspect_ratio(lm, LEFT_EYE) + eye_aspect_ratio(lm, RIGHT_EYE)) / 2
            if ear < 0.20:
                sleepy = True
            mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_TESSELATION)

    # -------- POSE --------
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if abs(l_sh.y - r_sh.y) > 0.05:
            bad_posture = True
        mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # -------- HAND --------
    if hand_res.multi_hand_landmarks:
        stressed = True
        for handLms in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # -------- SCORE --------
    score = 100
    if sleepy: score -= 30
    if distracted: score -= 20
    if bad_posture: score -= 15
    if stressed: score -= 10

    if score > 75:
        state = "Focused"
    elif score > 50:
        state = "Distracted"
    elif score > 30:
        state = "Tired / Stressed"
    else:
        state = "Highly Unfocused"

    st.session_state.logs.append([datetime.now(), state, score])

    cv2.putText(frame, f"{state} | Score: {score}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    st.image(frame, channels="BGR", use_container_width=True)
    st.metric("🎯 Focus State", state)
    st.metric("📊 Focus Score", score)

# --------------------
# Download report
# --------------------
if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs, columns=["Time", "State", "Score"])
    st.download_button("⬇ Download Session Report", df.to_csv(index=False), "focus_report.csv")

