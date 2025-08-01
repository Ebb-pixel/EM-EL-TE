import mmdet
import mmengine
import mmcv
import streamlit as st
import os
from mmdet.datasets.transforms import formatting  # Critical for registry
from mmpose.datasets.datasets import *  # For MMPose components
from inference_module import run_pose_estimation

st.title("🤿 Underwater Pose Estimation - RTMPose")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file is not None:
    st.video(video_file)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    input_path = os.path.join(output_dir, video_file.name)
    with open(input_path, "wb") as f:
        f.write(video_file.read())

    output_video = os.path.join(output_dir, f"output_{video_file.name}")
    output_json = os.path.join(output_dir, f"{os.path.splitext(video_file.name)[0]}.json")

    if st.button("Run Pose Estimation"):
        with st.spinner("Running..."):
            run_pose_estimation(input_path, output_video, output_json)
        st.success("Done!")
        st.video(output_video)
        with open(output_json, "r") as f:
            st.download_button("Download Keypoints JSON", f, file_name=os.path.basename(output_json))
