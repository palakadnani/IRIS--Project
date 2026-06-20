from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=416,
        batch=16,
        device=0,          # change to "cpu" if needed
        workers=0,         # safest on Windows; later you can try 2 or 4
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=5,
        translate=0.05,
        scale=0.3,
        mosaic=1.0,
        mixup=0.1,
        patience=10,
    )


if __name__ == "__main__":
    freeze_support()
    main()
