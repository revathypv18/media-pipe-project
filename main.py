import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from datetime import datetime
import math

# =========================
# INITIALIZE MEDIAPIPE
# =========================
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

face = mp_face.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

# =========================
# SESSION LOG
# =========================
logs = []

# =========================
# UTILS
# =========================
def distance(p1, p2):
    return math.dist((p1.x, p1.y), (p2.x, p2.y))

# Eye landmarks (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye):
    v1 = distance(landmarks[eye[1]], landmarks[eye[5]])
    v2 = distance(landmarks[eye[2]], landmarks[eye[4]])
    h = distance(landmarks[eye[0]], landmarks[eye[3]])
    return (v1 + v2) / (2.0 * h)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_res = face.process(rgb)
    pose_res = pose.process(rgb)
    hand_res = hands.process(rgb)

    sleepy = False
    distracted = False
    bad_posture = False
    stressed = False

    # =========================
    # FACE ANALYSIS
    # =========================
    if face_res.multi_face_landmarks:
        for fl in face_res.multi_face_landmarks:
            lm = fl.landmark

            left_ear = eye_aspect_ratio(lm, LEFT_EYE)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2

            if ear < 0.20:
                sleepy = True

            # Head tilt (distraction)
            nose = lm[1]
            left = lm[234]
            right = lm[454]

            if abs(left.y - right.y) > 0.03:
                distracted = True

            mp_draw.draw_landmarks(frame, fl, mp_face.FACEMESH_TESSELATION)

    # =========================
    # POSE ANALYSIS
    # =========================
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        l_sh = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_ear = lm[mp_pose.PoseLandmark.LEFT_EAR]
        r_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR]

        if abs(l_sh.y - r_sh.y) > 0.05 or abs(l_ear.y - l_sh.y) < 0.05:
            bad_posture = True

        mp_draw.draw_landmarks(frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # =========================
    # HAND ANALYSIS
    # =========================
    if hand_res.multi_hand_landmarks:
        stressed = True
        for handLms in hand_res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # =========================
    # FOCUS SCORING
    # =========================
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

    # =========================
    # LOGGING
    # =========================
    logs.append([datetime.now(), state, score])

    # =========================
    # DISPLAY
    # =========================
    cv2.putText(frame, f"State: {state}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame, f"Focus Score: {score}", (30,80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if sleepy:
        cv2.putText(frame, "Sleepy detected", (30,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    if bad_posture:
        cv2.putText(frame, "Bad posture", (30,160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    cv2.imshow("FocusSense - Emotion Aware Study Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# SAVE SESSION REPORT
# =========================
df = pd.DataFrame(logs, columns=["Time", "State", "FocusScore"])
df.to_csv("study_session_report.csv", index=False)

cap.release()
cv2.destroyAllWindows()

print("Session saved as study_session_report.csv")
print("Main file created successfully")

