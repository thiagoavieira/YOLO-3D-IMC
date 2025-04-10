import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from collections import deque

class Segmenter:
    """
    Segmentação de objetos com YOLOv11-seg da Ultralytics
    """
    def __init__(self, model_size='small', conf_thres=0.25, iou_thres=0.45, classes=None, device=None):
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

        print(f"Usando dispositivo: {self.device} para segmentação")

        # Nome do modelo
        model_map = {
            'nano': 'yolo11n-seg',
            'small': 'yolo11s-seg',
            'medium': 'yolo11m-seg',
            'large': 'yolo11l-seg',
            'extra': 'yolo11x-seg'
        }
        model_name = model_map.get(model_size.lower(), model_map['small'])

        # Carrega modelo
        self.model = YOLO(model_name)
        print(f"Modelo YOLOv11-seg ({model_size}) carregado.")

        # Configura parâmetros
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        if classes is not None:
            self.model.overrides['classes'] = classes

    def detect(self, image):
        """
        Detecta e segmenta objetos em uma imagem

        Args:
            image (numpy.ndarray): Imagem em BGR

        Returns:
            tuple: (imagem anotada, lista de detecções)
        """
        results = self.model.predict(image, verbose=False, device=self.device)
        annotated_image = image.copy()
        detections = []

        for prediction in results:
            if prediction is None or prediction.masks is None:
                continue

            boxes = prediction.boxes
            masks = prediction.masks.data.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            bboxes = boxes.xyxy.cpu().numpy()

            for i, mask in enumerate(masks):
                class_id = int(class_ids[i])
                score = float(scores[i])
                bbox = bboxes[i].tolist()

                # Aplica a máscara na imagem
                color = (0, 255, 0)
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_image, contours, -1, color, 2)

                # Caixa delimitadora
                xmin, ymin, xmax, ymax = map(int, bbox)
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                label = f"{prediction.names[class_id]} {score:.2f}"
                cv2.putText(annotated_image, label, (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                detections.append({
                    'bbox': bbox,
                    'score': score,
                    'class_id': class_id,
                    'mask': mask
                })

        return annotated_image, detections

    def get_class_names(self):
        return self.model.names
