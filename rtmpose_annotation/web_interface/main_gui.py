# pose_gui_static.py
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import threading
import os

CUSTOM_INFER_SCRIPT = '../mmpose/custom_video_pose_inference.py' # Update path if needed

OUTPUT_VIDEO = 'output/test_pose_output.avi'
OUTPUT_JSON = 'output/test_pose_output.json'

def run_inference(video_path, status_label):
    try:
        status_label.config(text="🔄 Running inference... Please wait.", fg="blue")
        subprocess.run(["python", CUSTOM_INFER_SCRIPT, video_path], check=True)
        status_label.config(text="✅ Inference complete!", fg="green")
        messagebox.showinfo("Done", f"Video saved to:\n{OUTPUT_VIDEO}\n\nKeypoints JSON:\n{OUTPUT_JSON}")
    except subprocess.CalledProcessError:
        status_label.config(text="❌ Inference failed.", fg="red")
        messagebox.showerror("Error", "Something went wrong during inference.")
    except Exception as e:
        status_label.config(text="❌ Unexpected error.", fg="red")
        messagebox.showerror("Error", str(e))

def select_video(status_label, entry):
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
    if file_path:
        entry.delete(0, tk.END)
        entry.insert(0, file_path)
        status_label.config(text="📂 Video selected. Ready to run.", fg="blue")

def start_inference(entry, status_label):
    video_path = entry.get()
    if not os.path.exists(video_path):
        messagebox.showwarning("Input Error", "Please select a valid video file.")
        return
    threading.Thread(target=run_inference, args=(video_path, status_label), daemon=True).start()

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Pose Estimation GUI")
root.geometry("520x300")
root.configure(bg="#f0f2f5")

title = tk.Label(root, text="🎯 Underwater Pose Estimation", font=("Helvetica", 16, "bold"), bg="#f0f2f5")
title.pack(pady=10)

frame = tk.Frame(root, bg="#ffffff", padx=20, pady=20, bd=2, relief=tk.RIDGE)
frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

entry_label = tk.Label(frame, text="Selected Video:", font=("Helvetica", 11), bg="#ffffff")
entry_label.pack(anchor="w")

entry = tk.Entry(frame, font=("Helvetica", 10), width=50)
entry.pack(pady=5)

select_btn = tk.Button(frame, text="📁 Browse", font=("Helvetica", 10), command=lambda: select_video(status_label, entry))
select_btn.pack()

run_btn = tk.Button(frame, text="▶ Run Pose Estimation", font=("Helvetica", 12, "bold"), bg="#0078D7", fg="white",
                    padx=10, pady=5, command=lambda: start_inference(entry, status_label))
run_btn.pack(pady=15)

status_label = tk.Label(root, text="Waiting for input...", fg="gray", bg="#f0f2f5", font=("Helvetica", 10, "italic"))
status_label.pack(pady=10)

root.mainloop()
