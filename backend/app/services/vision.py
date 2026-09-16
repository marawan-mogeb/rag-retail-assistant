from ultralytics import YOLO

from app.core.config import settings
from app.utils.logging_config import logger


class VisionService:
    """Loads the fine-tuned YOLO model once at startup and runs shelf detection."""

    def __init__(self):
        logger.info(f"Loading YOLO model from: {settings.yolo_model_path}")
        self.model = YOLO(settings.yolo_model_path)
        self.conf_threshold = settings.yolo_conf_threshold

    def detect(self, image_path: str) -> dict:
        results = self.model.predict(image_path, conf=self.conf_threshold, verbose=False)
        boxes = results[0].boxes
        num_products = len(boxes)
        confs = boxes.conf.tolist() if num_products > 0 else []
        return {
            "num_products_detected": num_products,
            "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        }
