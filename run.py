import os
import cv2
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator

# Configurações
root_dir = "/content/dataset_acamados_teste"  # Pasta raiz com subpastas de imagens
output_csv = "results.csv"
device = 'mps' # cpu or mps
yolo_model_size = "extra" # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
depth_model_size = "large" # Depth Anything v2 model size: "small", "base", "large"
conf_threshold = 0.25
iou_threshold = 0.45
classes = [0]  # None for all classes, otherwise [0]

def process_image(image_path, detector, depth_estimator, bbox3d_estimator):
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[WARNING] Não foi possível carregar imagem: {image_path}")
        return []

    height, width = frame.shape[:2]
    original_frame = frame.copy()

    # Detecção de objetos
    try:
        _, detections = detector.detect(frame, track=False)
    except Exception as e:
        print(f"[ERROR] Detecção falhou para {image_path}: {e}")
        return []

    # Estimativa de profundidade
    try:
        depth_map = depth_estimator.estimate_depth(original_frame)
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = normalized_depth.astype(np.uint8)
        cv2.imwrite(f"/content/depth_{Path(image_path).stem}.png", depth_vis)
    except Exception as e:
        print(f"[ERROR] Falha na profundidade para {image_path}: {e}")
        depth_map = np.zeros((height, width), dtype=np.float32)

    results = []

    for detection in detections:
        try:
            bbox, score, class_id, obj_id = detection
            class_name = detector.get_class_names()[class_id]

            if class_name.lower() in ['person', 'cat', 'dog']:
                x1, y1, x2, y2 = map(int, bbox)
                
                # Centro da bbox horizontal
                cx = int((x1 + x2) / 2)

                # Coordenadas verticais: topo (cabeça), meio, base (pés)
                head_y = y1 + int((y2 - y1) * 0.15)
                center_y = y1 + int((y2 - y1) * 0.5)
                feet_y = y1 + int((y2 - y1) * 0.85)
                depth_head = depth_estimator.get_depth_at_point(depth_map, cx, head_y)
                depth_center = depth_estimator.get_depth_at_point(depth_map, cx, center_y)
                depth_feet = depth_estimator.get_depth_at_point(depth_map, cx, feet_y)
                
                method = "multi-point"
            else:
                depth_value_median = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                method = "median"

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
                "object_id": obj_id
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
    depth_estimator = DepthEstimator(model_size=depth_model_size, device=device)
    bbox3d_estimator = BBox3DEstimator()

    all_results = []

    for idx, image_path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] Processando {image_path}...")
        results = process_image(image_path, detector, depth_estimator, bbox3d_estimator)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"Processamento finalizado. Resultados salvos em: {output_csv}")

if __name__ == "__main__":
    main()
