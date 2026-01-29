import streamlit as st
import pandas as pd

st.set_page_config(page_title="FocusSense", layout="centered")

st.title("🎯 FocusSense – AI Based Study Focus Monitor")

st.markdown("""
### 📌 Project Overview
FocusSense is an AI-based system that monitors a student's focus level using:
- Face landmarks (eye blink, head tilt)
- Body posture analysis
- Stress indicators from hand movement

The real-time MediaPipe system is implemented and tested locally.
This web app demonstrates the system workflow, reporting, and analysis.
""")

st.subheader("🧠 System Capabilities")
st.write("""
✅ Drowsiness detection  
✅ Distraction analysis  
✅ Posture monitoring  
✅ Stress indication  
✅ Focus score generation  
✅ Session report generation  
""")

st.subheader("⚙️ Technologies Used")
st.code("""
Python
MediaPipe
OpenCV
Streamlit
NumPy
Pandas
""")

st.subheader("📊 Sample Session Report")

try:
    df = pd.read_csv("study_session_report.csv")
    st.dataframe(df)

    st.download_button(
        "⬇ Download Sample Report",
        df.to_csv(index=False),
        "focus_report.csv"
    )
except:
    st.info("Run the system locally to generate study_session_report.csv")

st.subheader("📷 System Demonstration")
st.info("Live webcam processing is performed locally due to current cloud limitations of OpenCV & MediaPipe.")

st.success("✔ Project deployed successfully and connected to GitHub.")

