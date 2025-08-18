# inference/gui_inference.py
import os, sys, threading, traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ------------- bundled-file helpers -------------
def rbase():
    # folder of this file (script) or PyInstaller’s temp (exe)
    return getattr(sys, "_MEIPASS", os.path.dirname(__file__))

def rpath(*parts):
    return os.path.join(rbase(), *parts)

# ------------- core inference worker -------------
def run_inference(video_path, use_gpu, smooth, progress_cb, done_cb):
    """
    Heavy worker that runs in a background thread.
    Calls:
      progress_cb(done_frames, total_frames, status_text)
      done_cb(success: bool, message: str, out_dir: str|None)
    """
    try:
        # --- lazy import heavy deps inside the worker ---
        import json, time
        import cv2
        import numpy as np
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet.apis import init_detector, inference_detector
        from mmpose.apis import init_model as init_pose_model
        from mmpose.apis import inference_topdown
        from mmpose.visualization import PoseLocalVisualizer

        # --- resolve bundled assets ---
        DET_CFG   = rpath("configs", "rtmdet_tiny_infer_flat.py")
        DET_CKPT  = rpath("models",  "rtmdet.pth")
        POSE_CFG  = rpath("configs", "rtmpose-l_infer_flat.py")
        POSE_CKPT = rpath("models",  "rtmpose.pth")
        for p in (DET_CFG, DET_CKPT, POSE_CFG, POSE_CKPT):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Bundled file missing: {p}")

        # --- output paths ---
        video_path = os.path.abspath(video_path)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Input video not found:\n{video_path}")
        out_dir = os.path.join(os.path.dirname(video_path), "pose_out")
        os.makedirs(out_dir, exist_ok=True)
        out_video_path = os.path.join(out_dir, "output_pose.mp4")
        out_json_path  = os.path.join(out_dir, "output_keypoints.json")

        device = "cuda" if use_gpu else "cpu"
        progress_cb(0, 0, "Loading models...")

        # --- init models ---
        init_default_scope("mmdet")
        det_model  = init_detector(DET_CFG,  DET_CKPT,  device=device)
        init_default_scope("mmpose")
        pose_model = init_pose_model(POSE_CFG, POSE_CKPT, device=device)

        # --- dataset meta for visualizer ---
        pose_cfg = Config.fromfile(POSE_CFG)
        dataset_meta = pose_cfg.get("dataset_info", {}) or {}
        if "skeleton_info" in dataset_meta and "keypoint_info" in dataset_meta:
            kp_name_to_id = {v["name"]: int(k) for k, v in dataset_meta["keypoint_info"].items()}
            dataset_meta["skeleton"] = [
                [kp_name_to_id[c["link"][0]], kp_name_to_id[c["link"][1]]]
                for c in dataset_meta["skeleton_info"].values()
            ]
            dataset_meta["pose_kpt_color"]  = [v["color"] for v in dataset_meta["keypoint_info"].values()]
            dataset_meta["pose_link_color"] = [c["color"] for c in dataset_meta["skeleton_info"].values()]
            dataset_meta["num_keypoints"]   = len(dataset_meta["keypoint_info"])

        visualizer = PoseLocalVisualizer(line_width=3, radius=4)
        visualizer.set_dataset_meta(dataset_meta)
        pose_model.dataset_meta = dataset_meta

        # --- video IO ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_vid= cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        # --- smoothing setup ---
        alpha = 0.4 if smooth else None
        smoothed = {}
        results_json = []
        done = 0
        import time as _t
        start_t = _t.time()

        progress_cb(0, max(total, 1), f"Processing... 0/{total if total>0 else '?'}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            done += 1

            # Detection
            init_default_scope("mmdet")
            det_res = inference_detector(det_model, frame)
            bboxes  = det_res.pred_instances.bboxes.cpu().numpy()
            scores  = det_res.pred_instances.scores.cpu().numpy()
            labels  = det_res.pred_instances.labels.cpu().numpy()

            H, W = frame.shape[:2]
            def expand_bbox(b, W, H, scale=1.2):
                x1, y1, x2, y2 = b
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w/2, y1 + h/2
                nw, nh = w*scale, h*scale
                nx1 = max(0, int(cx - nw/2)); ny1 = max(0, int(cy - nh/2))
                nx2 = min(W-1, int(cx + nw/2)); ny2 = min(H-1, int(cy + nh/2))
                return [nx1, ny1, nx2, ny2]

            person_boxes = [expand_bbox(b, W, H, 1.2)
                            for b, s, lab in zip(bboxes, scores, labels) if lab == 0 and s > 0.5]

            if not person_boxes:
                out_vid.write(frame)
                results_json.append({"frame_id": done, "instances": []})
                progress_cb(done, max(total, 1), f"Processing... {done}/{total if total>0 else '?'}")
                continue

            # Pose
            init_default_scope("mmpose")
            pose_results = inference_topdown(
                pose_model, frame, np.array(person_boxes, dtype=np.float32)
            )

            if alpha is not None:
                for i, pr in enumerate(pose_results):
                    k = pr.pred_instances.keypoints
                    prev = smoothed.get(i)
                    smoothed[i] = k if prev is None else (alpha * k + (1 - alpha) * prev)
                    pr.pred_instances.keypoints = smoothed[i]

            # Visualize
            visualizer.set_image(frame.copy())
            for pr in pose_results:
                visualizer.add_datasample(
                    name="result",
                    image=frame,
                    data_sample=pr,
                    draw_gt=False,
                    draw_pred=True,
                    kpt_thr=0.25,
                    show=False
                )
            vis = visualizer.get_image()
            for (x1, y1, x2, y2) in person_boxes:
                import cv2 as _cv
                _cv.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            out_vid.write(vis)

            # JSON
            instances = []
            for pr, raw_box in zip(pose_results, person_boxes):
                inst = {"bbox": list(map(float, raw_box)),
                        "keypoints": pr.pred_instances.keypoints.tolist()}
                if hasattr(pr.pred_instances, "keypoint_scores"):
                    inst["scores"] = pr.pred_instances.keypoint_scores.tolist()
                instances.append(inst)
            results_json.append({"frame_id": done, "instances": instances})

            progress_cb(done, max(total, 1), f"Processing... {done}/{total if total>0 else '?'}")

        cap.release()
        out_vid.release()
        with open(out_json_path, "w") as f:
            json.dump(results_json, f, indent=2)

        elapsed = _t.time() - start_t
        msg = f"Saved:\n• {out_video_path}\n• {out_json_path}\nTime: {elapsed:.1f}s"
        done_cb(True, msg, out_dir)
    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        done_cb(False, f"Error: {err}", None)

# ------------- Tkinter GUI -------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Underwater Pose Estimation")
        self.geometry("560x260")
        self.resizable(False, False)

        # styles
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # widgets
        pad = {"padx": 12, "pady": 6}

        self.path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Select a video to begin.")
        self.total_frames = 0

        row = 0
        ttk.Label(self, text="Input video:").grid(row=row, column=0, sticky="w", **pad)
        e = ttk.Entry(self, textvariable=self.path_var, width=55)
        e.grid(row=row, column=1, sticky="we", **pad)
        ttk.Button(self, text="Browse...", command=self.browse).grid(row=row, column=2, **pad)

        row += 1
        self.use_gpu = tk.BooleanVar(value=False)
        self.smooth  = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Use GPU (CUDA)", variable=self.use_gpu).grid(row=row, column=1, sticky="w", **pad)
        ttk.Checkbutton(self, text="Temporal smoothing", variable=self.smooth).grid(row=row, column=2, sticky="w", **pad)

        row += 1
        self.progress = ttk.Progressbar(self, orient="horizontal", length=420, mode="determinate", maximum=100)
        self.progress.grid(row=row, column=0, columnspan=3, sticky="we", padx=12, pady=(12, 6))

        row += 1
        ttk.Label(self, textvariable=self.status_var, wraplength=520, justify="left").grid(
            row=row, column=0, columnspan=3, sticky="w", padx=12
        )

        row += 1
        self.start_btn = ttk.Button(self, text="Start", command=self.start)
        self.start_btn.grid(row=row, column=2, sticky="e", padx=12, pady=8)

        # open file dialog immediately on launch
        self.after(200, self.browse)

    def browse(self):
        path = filedialog.askopenfilename(
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"), ("All files", "*.*")]
        )
        if path:
            self.path_var.set(path)
            self.status_var.set("Ready. Click Start to process.")

    def start(self):
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("No video", "Please choose a video file.")
            return
        if not os.path.isfile(path):
            messagebox.showerror("Not found", f"File not found:\n{path}")
            return

        # lock UI
        self.start_btn.configure(state="disabled")
        self.status_var.set("Initializing...")
        self.progress.configure(value=0)

        # spawn worker thread
        t = threading.Thread(
            target=run_inference,
            args=(path, self.use_gpu.get(), self.smooth.get(), self.on_progress, self.on_done),
            daemon=True
        )
        t.start()

    def on_progress(self, done, total, status_text):
        # called from worker thread -> marshal to UI thread
        def _upd():
            pct = int(done * 100 / total) if total > 0 else 0
            self.progress.configure(value=pct)
            self.status_var.set(status_text)
        self.after(0, _upd)

    def on_done(self, ok, message, out_dir):
        # called from worker thread -> marshal to UI thread
        def _fin():
            self.start_btn.configure(state="normal")
            self.progress.configure(value=100 if ok else 0)
            self.status_var.set("Completed." if ok else "Failed.")
            if ok:
                messagebox.showinfo("Done", message)
                # Optionally, open the folder
                try:
                    if out_dir and os.path.isdir(out_dir):
                        os.startfile(out_dir)  # Windows only
                except Exception:
                    pass
            else:
                messagebox.showerror("Error", message)
        self.after(0, _fin)

if __name__ == "__main__":
    # GUI app only; no console interaction
    app = App()
    app.mainloop()
