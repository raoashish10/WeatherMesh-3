import pipeline  # noqa: F401  (adds repo root to sys.path)
import torch
from model import get_WeatherMesh3


def load_model(weights_path="weights/WeatherMesh3.pt", device="cuda"):
    model = get_WeatherMesh3(weights_path=weights_path)
    model = model.to(device).eval()
    return model
