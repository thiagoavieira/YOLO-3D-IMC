import os
import cv2
import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from detection_model import ObjectDetector
from segmentation_model import Segmenter  # novo import
from pose_estimation_model import PoseEstimator # novo import
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator

# Configurações
root_dir = "/dir"  # Pasta raiz com subpastas de imagens
output_csv = "results.csv"
device = 'cuda' # cpu, cuda or mps
yolo_model_size = "extra" # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
depth_model_size = "large" # Depth Anything v2 model size: "small", "base", "large"
conf_threshold = 0.25
iou_threshold = 0.45
classes = [0]  # None for all classes, otherwise [0]

# Ativar segmentacao
use_segmentation = True

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def process_image(image_path, detector, segmenter, depth_estimator, bbox3d_estimator, pose_estimator):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[WARNING] Não foi possível carregar imagem: {image_path}")
        return []

    height, width = frame.shape[:2]
    original_frame = frame.copy()

    try:
        if segmenter:
            _, detections = segmenter.detect(frame)
        else:
            _, detections = detector.detect(frame, track=False)
    except Exception as e:
        print(f"[ERROR] Detecção falhou para {image_path}: {e}")
        return []

    try:
        depth_map = depth_estimator.estimate_depth(original_frame)
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = normalized_depth.astype(np.uint8)
        cv2.imwrite(f"/content/depth_{Path(image_path).stem}.png", depth_vis)
    except Exception as e:
        print(f"[ERROR] Falha na profundidade para {image_path}: {e}")
        depth_map = np.zeros((height, width), dtype=np.float32)

    try:
        _, pose_detections = pose_estimator.detect(frame)
    except Exception as e:
        print(f"[ERROR] Falha no pose estimation para {image_path}: {e}")

    results = []

    for detection in detections:
        try:
            if segmenter:
                segmentation = detection.get('mask', None)
                bbox = detection['bbox']
                score = detection['score']
                class_id = detection['class_id']
                class_name = segmenter.get_class_names()[class_id]
                obj_id = -1
                if isinstance(segmentation, np.ndarray):
                    segmentation = segmentation.tolist()
                segmentation_json = json.dumps(segmentation) if segmentation is not None else None
            else:
                bbox, score, class_id, obj_id = detection
                class_name = detector.get_class_names()[class_id]
                segmentation_json = None

            if class_name.lower() in ['person', 'cat', 'dog']:
                x1, y1, x2, y2 = map(int, bbox)
                cx = int((x1 + x2) / 2)
                head_y = y1 + int((y2 - y1) * 0.15)
                center_y = y1 + int((y2 - y1) * 0.5)
                feet_y = y1 + int((y2 - y1) * 0.85)
                depth_head = depth_estimator.get_depth_at_point(depth_map, cx, head_y)
                depth_center = depth_estimator.get_depth_at_point(depth_map, cx, center_y)
                depth_feet = depth_estimator.get_depth_at_point(depth_map, cx, feet_y)
                method = "multi-point"
            else:
                depth_head = depth_center = depth_feet = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                method = "median"

            # Buscar keypoints que batem com a bbox
            keypoints = None
            for pose_det in pose_detections:
                px1, py1, px2, py2 = map(int, pose_det['bbox'])
                iou = bbox_iou((x1, y1, x2, y2), (px1, py1, px2, py2))
                if iou > iou_threshold:  # threshold arbitrário
                    keypoints = pose_det.get('keypoints', None)
                    break

            keypoints_json = json.dumps(keypoints) if keypoints is not None else None

            results.append({
                "image": str(image_path),
                "class": class_name,
                "score": float(score),
                "bbox_x1": int(bbox[0]),
                "bbox_y1": int(bbox[1]),
                "bbox_x2": int(bbox[2]),
                "bbox_y2": int(bbox[3]),
                "depth_head": float(depth_head),
                "depth_center": float(depth_center),
                "depth_feet": float(depth_feet),
                "depth_method": method,
                "object_id": obj_id,
                "segmentation": segmentation_json,
                "keypoints": keypoints_json  # NOVO CAMPO
            })
        except Exception as e:
            print(f"[ERROR] Erro ao processar detecção em {image_path}: {e}")
            continue

    return results

def main():
    print(f"Procurando imagens em {root_dir}...")
    image_paths = list(Path(root_dir).rglob("*.[jp][pn]g"))
    print(f"{len(image_paths)} imagens encontradas.")

    detector = ObjectDetector(
        model_size=yolo_model_size,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        classes=classes,
        device=device
    )

    segmenter = Segmenter(
        model_size=yolo_model_size,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,
        classes=classes,
        device=device
    ) if use_segmentation else None

    depth_estimator = DepthEstimator(model_size=depth_model_size, device=device)
    bbox3d_estimator = BBox3DEstimator()
    pose_estimator = PoseEstimator(model_size=yolo_model_size, device=device)

    all_results = []
    for idx, image_path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] Processando {image_path}...")
        results = process_image(image_path, detector, segmenter, depth_estimator, bbox3d_estimator, pose_estimator)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Processamento finalizado. Resultados salvos em: {output_csv}")

if __name__ == "__main__":
    main()
