# inference/custom_inference.py
import os, sys, argparse, json, time
import cv2
import numpy as np

# ---------- Helpers for bundled exe ----------
def rbase():
    """Runtime base: folder of this file (script) or PyInstaller's temp (exe)."""
    return getattr(sys, "_MEIPASS", os.path.dirname(__file__))

def rpath(*parts):
    return os.path.join(rbase(), *parts)

def parse_args():
    ap = argparse.ArgumentParser(
        description="Underwater swimmer detection + pose (single-file exe, console UI)."
    )
    # --input optional: if missing we show a file chooser
    ap.add_argument("--input", default=None, help="Path to input video file")
    ap.add_argument("--output-dir", default=None, help="Folder to save outputs (default: <input_dir>/pose_out)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device")
    ap.add_argument("--smooth", action="store_true", help="Enable temporal smoothing (EMA)")
    return ap.parse_args()

def choose_video_with_gui():
    """Open a file chooser and force it to the front; fall back to empty string on error."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        # make the dialog come to the front
        root.attributes('-topmost', True)
        root.update()

        path = filedialog.askopenfilename(
            parent=root,
            title="Select input video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"),
                       ("All files", "*.*")]
        )
        root.destroy()
        return path or ""
    except Exception:
        return ""

def notify_done(title, message):
    """Prefer a Windows 10+ toast; fall back to Tk popup; else print."""
    # 1) Windows toast
    try:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        # duration is seconds; threaded=True returns immediately
        toaster.show_toast(title, message, duration=6, threaded=True)
        return
    except Exception:
        pass

    # 2) Tk popup
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, message)
        root.destroy()
        return
    except Exception:
        pass

    # 3) Console
    print(f"\n*** {title}: {message} ***")

def expand_bbox(bbox, W, H, scale=1.2):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    nw, nh = w * scale, h * scale
    nx1 = max(0, int(cx - nw / 2)); ny1 = max(0, int(cy - nh / 2))
    nx2 = min(W - 1, int(cx + nw / 2)); ny2 = min(H - 1, int(cy + nh / 2))
    return [nx1, ny1, nx2, ny2]

def print_progress(done, total, start_time):
    """Simple console progress bar without extra deps."""
    if total <= 0:
        # unknown length; print dots occasionally
        if done % 30 == 0:
            print(".", end="", flush=True)
        return
    pct = int((done / total) * 100)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    msg = f"\r[{('#'*filled).ljust(bar_len)}] {pct:3d}%  {done}/{total} frames  ({rate:.1f} fps)"
    print(msg, end="", flush=True)

def main():
    args = parse_args()

    # -------- Pick input --------
    if args.input:
        in_path = args.input
    else:
        print("📂 No --input provided. Opening file chooser... (If nothing appears, press Alt+Tab.)", flush=True)
        in_path = choose_video_with_gui()
        if not in_path:
            # Console fallback if the file dialog didn't appear or was cancelled
            in_path = input("Enter full path to the input video (or press Enter to exit): ").strip()
            if not in_path:
                print("[ERROR] No input video selected/provided.")
                sys.exit(1)

    # -------- Output folder --------
    out_dir = args.output_dir or os.path.join(os.path.dirname(in_path), "pose_out")
    os.makedirs(out_dir, exist_ok=True)
    out_video_path = os.path.join(out_dir, "output_pose.mp4")
    out_json_path  = os.path.join(out_dir, "output_keypoints.json")

    # -------- Bundled assets (added via PyInstaller --add-data) --------
    DET_CFG   = rpath("configs", "rtmdet_tiny_infer_flat.py")
    DET_CKPT  = rpath("models",  "rtmdet.pth")
    POSE_CFG  = rpath("configs", "rtmpose-l_infer_flat.py")
    POSE_CKPT = rpath("models",  "rtmpose.pth")

    for p in (DET_CFG, DET_CKPT, POSE_CFG, POSE_CKPT):
        if not os.path.isfile(p):
            print(f"[ERROR] Bundled file missing: {p}")
            sys.exit(2)

    # -------- Delay heavy imports until after CLI parsing --------
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmdet.apis import init_detector, inference_detector
    from mmpose.apis import init_model as init_pose_model
    from mmpose.apis import inference_topdown
    from mmpose.visualization import PoseLocalVisualizer

    print("🔧 Loading models (this can take a moment)...")
    device = args.device
    init_default_scope("mmdet")
    det_model = init_detector(DET_CFG, DET_CKPT, device=device)

    init_default_scope("mmpose")
    pose_model = init_pose_model(POSE_CFG, POSE_CKPT, device=device)

    pose_cfg = Config.fromfile(POSE_CFG)
    dataset_meta = pose_cfg.get("dataset_info", {}) or {}
    # Convert skeleton_info/keypoint_info to visualizer-friendly fields if needed
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

    # -------- Video I/O --------
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {in_path}")
        sys.exit(3)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid= cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    print(f"🎬 Input: {os.path.basename(in_path)}  ({width}x{height} @ {fps:.1f} fps)")
    if total > 0:
        print(f"📏 Frames: {total}")
    print(f"💾 Output folder: {out_dir}")
    print("🚀 Processing...")

    alpha = 0.4 if args.smooth else None
    smoothed = {}
    results_json = []
    frame_id = 0
    start_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        # Detection
        init_default_scope("mmdet")
        det_res = inference_detector(det_model, frame)
        bboxes  = det_res.pred_instances.bboxes.cpu().numpy()
        scores  = det_res.pred_instances.scores.cpu().numpy()
        labels  = det_res.pred_instances.labels.cpu().numpy()

        H, W = frame.shape[:2]
        person_boxes = [expand_bbox(b, W, H, 1.2)
                        for b, s, lab in zip(bboxes, scores, labels) if lab == 0 and s > 0.5]

        if not person_boxes:
            out_vid.write(frame)
            results_json.append({"frame_id": frame_id, "instances": []})
            print_progress(frame_id, total, start_t)
            continue

        # Pose
        init_default_scope("mmpose")
        pose_results = inference_topdown(pose_model, frame, np.array(person_boxes, dtype=np.float32))

        # Smoothing (EMA per index)
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
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        out_vid.write(vis)

        # JSON
        instances = []
        for pr, raw_box in zip(pose_results, person_boxes):
            inst = {
                "bbox": list(map(float, raw_box)),
                "keypoints": pr.pred_instances.keypoints.tolist()
            }
            if hasattr(pr.pred_instances, "keypoint_scores"):
                inst["scores"] = pr.pred_instances.keypoint_scores.tolist()
            instances.append(inst)
        results_json.append({"frame_id": frame_id, "instances": instances})

        print_progress(frame_id, total, start_t)

    cap.release()
    out_vid.release()
    with open(out_json_path, "w") as f:
        json.dump(results_json, f, indent=2)

    elapsed = time.time() - start_t
    print("\n✅ Done!")
    print(f"   ▶ Saved video: {out_video_path}")
    print(f"   🧩 Saved JSON:  {out_json_path}")
    print(f"   ⏱  Time: {elapsed:.1f}s  |  Avg FPS: {frame_id/elapsed if elapsed>0 else 0:.1f}")

    # Notification
    notify_done("Underwater Pose Estimation", f"Finished. Saved to:\n{out_dir}")

if __name__ == "__main__":
    main()
