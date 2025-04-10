import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO

class PoseEstimator:
    """
    Estimação de pose com YOLOv11-pose da Ultralytics
    """
    def __init__(self, model_size='extra', conf_thres=0.25, iou_thres=0.45, device=None):
        # Determina dispositivo
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device

        if self.device == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            print("Usando MPS com fallback para CPU")

        print(f"Usando dispositivo: {self.device} para pose estimation")

        # Nome do modelo
        model_map = {
            'nano': 'yolo11n-pose',
            'small': 'yolo11s-pose',
            'medium': 'yolo11m-pose',
            'large': 'yolo11l-pose',
            'extra': 'yolo11x-pose'
        }
        model_name = model_map.get(model_size.lower(), model_map['extra'])

        # Carrega modelo
        self.model = YOLO(model_name)
        print(f"Modelo YOLOv11-pose ({model_size}) carregado.")

        # Configura parâmetros
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000

    def detect(self, image):
        """
        Detecta keypoints em uma imagem

        Args:
            image (numpy.ndarray): Imagem em BGR

        Returns:
            tuple: (imagem anotada, lista de detecções com keypoints)
        """
        results = self.model.predict(image, verbose=False, device=self.device)
        annotated_image = image.copy()
        detections = []

        for prediction in results:
            boxes = prediction.boxes
            kpts = prediction.keypoints

            if boxes is None or kpts is None:
                continue

            bboxes = boxes.xyxy.cpu().numpy()
            keypoints = kpts.xy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()

            for i, bbox in enumerate(bboxes):
                class_id = int(class_ids[i])
                score = float(scores[i])
                pts = keypoints[i]  # (17, 2)

                # Caixa delimitadora
                xmin, ymin, xmax, ymax = map(int, bbox)
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                # Keypoints
                for (x, y) in pts:
                    if x > 0 and y > 0:
                        cv2.circle(annotated_image, (int(x), int(y)), 4, (0, 255, 255), -1)

                label = f"{prediction.names[class_id]} {score:.2f}"
                cv2.putText(annotated_image, label, (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    'bbox': bbox.tolist(),
                    'score': score,
                    'class_id': class_id,
                    'keypoints': pts.tolist()
                })

        return annotated_image, detections

    def get_class_names(self):
        return self.model.names
