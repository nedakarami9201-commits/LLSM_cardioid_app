import streamlit as st
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("Video–Trace Sync Viewer")

# Upload files
video_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
csv_file   = st.file_uploader("Upload analysis CSV", type=["csv"])

if video_file and csv_file:
    # --- Load analysis
    df = pd.read_csv(csv_file)
    if not {"time_s","voltage_like","contraction_like"}.issubset(df.columns):
        st.error("CSV must have columns: time_s, voltage_like, contraction_like")
    else:
        t = df["time_s"].values
        v = df["voltage_like"].values
        c = df["contraction_like"].values

        # --- Video setup
        tmpfile = f"/tmp/{video_file.name}"
        with open(tmpfile, "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture(tmpfile)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

        # Mapping time <-> frames
        frame_from_time = lambda tt: int(np.clip((tt - t[0])/(t[-1]-t[0])*(n_frames-1),0,n_frames-1))

        # --- Layout
        col1, col2 = st.columns([1,2])

        # Slider for time
        time_val = st.slider("Time (s)", float(t[0]), float(t[-1]), float(t[0]), step=float((t[-1]-t[0])/len(t)))

        # Video frame
        f_idx = frame_from_time(time_val)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            col1.image(frame, caption=f"Frame {f_idx}/{n_frames}", use_column_width=True)

        # Plot traces
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(t,v,label="Voltage-like")
        ax.plot(t,c,label="Contraction-like")
        ax.axvline(time_val,color="k",ls="--")
        ax.set_xlabel("Time (s)")
        ax.legend()
        col2.pyplot(fig)

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
