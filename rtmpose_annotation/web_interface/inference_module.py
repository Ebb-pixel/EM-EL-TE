# inference_module.py
import mmdet
import mmengine
import mmcv
# Add this early in your inference_module.py
from mmdet.datasets import *  # Force registry init
from mmdet.models import *    # Include models and transforms
from mmdet.structures import *  # Needed for PackDetInputs
import cv2, json, numpy as np
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.models import build_pose_estimator
from mmpose.visualization import PoseLocalVisualizer
from collections import defaultdict
from mmengine import DefaultScope
from mmpose.utils import register_all_modules


                    
def run_pose_estimation(video_path, output_video_path, output_json_path):
    
    # configs
    det_config = '../mmdetection/work_dirs/rtmdet_tiny_1class_underwater/rtmdet_tiny_1class_underwater.py'
    det_checkpoint = '../mmdetection/work_dirs/rtmdet_tiny_1class_underwater/best_coco_bbox_mAP_epoch_17.pth'
    pose_config = '../mmpose/configs/underwater/rtmpose-l_underwater.py'
    pose_checkpoint = '../mmpose/work_dirs/rtmpose-l_underwater/best_coco_AP_epoch_30.pth'
    device = 'cuda:0'

    DefaultScope.get_instance("mmdet_scope", scope_name="mmdet")
    detector = init_detector(det_config, det_checkpoint, device=device)

    
    pose_cfg = Config.fromfile(pose_config)
    register_all_modules()  # This registers TopdownPoseEstimator and others
    pose_model = build_pose_estimator(pose_cfg.model)
    pose_model = build_pose_estimator(pose_cfg.model)
    load_checkpoint(pose_model, pose_checkpoint, map_location='cpu')
    pose_model.cfg = pose_cfg
    pose_model.eval()

    dataset_meta = pose_cfg.dataset_info
    keypoint_name_to_id = {v['name']: int(k) for k, v in dataset_meta['keypoint_info'].items()}
    dataset_meta['skeleton'] = [
        [keypoint_name_to_id[conn['link'][0]], keypoint_name_to_id[conn['link'][1]]]
        for conn in dataset_meta['skeleton_info'].values()
    ]
    dataset_meta['pose_kpt_color'] = [v['color'] for v in dataset_meta['keypoint_info'].values()]
    dataset_meta['pose_link_color'] = [conn['color'] for conn in dataset_meta['skeleton_info'].values()]
    dataset_meta['num_keypoints'] = len(dataset_meta['keypoint_info'])

    visualizer = PoseLocalVisualizer(line_width=3, radius=4)
    visualizer.set_dataset_meta(dataset_meta)
    pose_model.dataset_meta = dataset_meta

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_id = 0
    results_json = []
    smoothed_kpts = defaultdict(lambda: None)
    alpha = 0.4

    def expand_bbox(bbox, image_shape, scale=0.2):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        new_w, new_h = w * scale, h * scale
        return [
            max(0, int(cx - new_w / 2)),
            max(0, int(cy - new_h / 2)),
            min(image_shape[1] - 1, int(cx + new_w / 2)),
            min(image_shape[0] - 1, int(cy + new_h / 2)),
        ]

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_id += 1

        DefaultScope.get_instance("mmdet_scope", scope_name="mmdet")
        det_result = inference_detector(detector, frame)
        bboxes = det_result.pred_instances.bboxes.cpu().numpy()
        scores = det_result.pred_instances.scores.cpu().numpy()
        labels = det_result.pred_instances.labels.cpu().numpy()

        person_bboxes = [bbox for bbox, score, label in zip(bboxes, scores, labels) if label == 0 and score > 0.5]
        if not person_bboxes:
            out_video.write(frame)
            continue

        bboxes_np = np.array([expand_bbox(b, frame.shape, scale=1.2) for b in person_bboxes], dtype=np.float32)

        DefaultScope.get_instance("mmpose_scope", scope_name="mmpose")
        pose_results = inference_topdown(pose_model, frame, bboxes_np)

        vis_frame = frame.copy()
        visualizer.set_image(vis_frame)
        for i, pose_result in enumerate(pose_results):
            kpts = pose_result.pred_instances.keypoints
            if smoothed_kpts[i] is None:
                smoothed_kpts[i] = kpts
            else:
                smoothed_kpts[i] = alpha * kpts + (1 - alpha) * smoothed_kpts[i]
            pose_result.pred_instances.keypoints = smoothed_kpts[i]

            visualizer.add_datasample(
                name='result',
                image=vis_frame,
                data_sample=pose_result,
                draw_gt=False,
                draw_pred=True,
                kpt_thr=0.25,
                show=False
            )
        out_video.write(visualizer.get_image())

        results_json.append({
            'frame_id': frame_id,
            'instances': [{
                'keypoints': r.pred_instances.keypoints.tolist(),
                'bbox': b.tolist()
            } for r, b in zip(pose_results, person_bboxes)]
        })

    cap.release()
    out_video.release()
    with open(output_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
