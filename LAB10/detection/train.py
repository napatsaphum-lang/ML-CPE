from ultralytics import YOLO
from .data_loader import ROOT

EXPERIMENTS = {
    'A': dict(epochs=10, imgsz=416, batch=8),
    'B': dict(epochs=20, imgsz=416, batch=8),
    'C': dict(epochs=30, imgsz=416, batch=8),
    'D': dict(epochs=20, imgsz=640, batch=8),
    'E': dict(epochs=20, imgsz=416, batch=16),
}

def train_one(data_yaml, name, quick=False):
    settings = EXPERIMENTS[name].copy()
    if quick:
        settings.update(epochs=1, imgsz=320, batch=4)
    model = YOLO('yolov8n.pt')  # fresh pretrained weights for each experiment
    model.train(
        data=str(data_yaml), project=str(ROOT/'outputs'), name=f'model_{name}' + ('_quick' if quick else ''),
        exist_ok=True, plots=True, seed=42, workers=0, device='cpu', **settings
    )
    return ROOT/'outputs'/(f'model_{name}' + ('_quick' if quick else ''))/'weights'/'best.pt'
