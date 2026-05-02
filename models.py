import os
import threading

import torch
import torch.nn as nn
import torchvision.models as tv_models

import config


class ChestNet(nn.Module):
    """ResNet-50 backbone adapted for 7-class chest pathology classification."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        backbone = tv_models.resnet50(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelRegistry:
    def __init__(self):
        self.device = _select_device()
        self.global_model: ChestNet | None = None
        self.client_models: list[ChestNet] = []
        self.client_paths: list[str] = []
        self.global_path: str | None = None
        self.loaded = False
        self._lock = threading.Lock()

    def _load_one(self, path: str) -> ChestNet:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        model = ChestNet(num_classes=len(config.CHEST_CLASSES)).to(self.device)
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def load(self):
        with self._lock:
            if self.loaded:
                return
            model_dir = os.path.join(
                os.path.dirname(config.BASE_DIR), "fl_outputs", "models"
            )
            global_path = os.path.join(model_dir, "global_model.pth")
            self.global_model = self._load_one(global_path)
            self.global_path = global_path
            self.client_models = []
            self.client_paths = []
            for i in range(config.NUM_CLIENTS):
                path = os.path.join(model_dir, f"final_client_{i}.pth")
                self.client_models.append(self._load_one(path))
                self.client_paths.append(path)
            self.loaded = True

    @torch.no_grad()
    def _infer(self, model: ChestNet, tensor: torch.Tensor) -> dict:
        tensor = tensor.to(self.device)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(probs.argmax())
        return {
            "predicted_index": pred_idx,
            "predicted_label": config.CHEST_CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": [float(p) for p in probs],
        }

    def predict_all(self, tensor: torch.Tensor) -> dict:
        if not self.loaded:
            self.load()
        results = {
            "global": {
                "name": "Global (FedAvg)",
                "source": os.path.basename(self.global_path),
                **self._infer(self.global_model, tensor),
            },
            "clients": [],
        }
        for i, model in enumerate(self.client_models):
            results["clients"].append({
                "name": f"Client {i}",
                "source": os.path.basename(self.client_paths[i]),
                **self._infer(model, tensor),
            })
        votes = [results["global"]["predicted_index"]] + [
            c["predicted_index"] for c in results["clients"]
        ]
        consensus_idx = max(set(votes), key=votes.count)
        agreement = votes.count(consensus_idx)
        results["consensus"] = {
            "predicted_index": consensus_idx,
            "predicted_label": config.CHEST_CLASSES[consensus_idx],
            "agreement": agreement,
            "total_models": len(votes),
            "unanimous": agreement == len(votes),
        }
        return results

    def info(self) -> dict:
        return {
            "device": str(self.device),
            "num_clients": config.NUM_CLIENTS,
            "num_classes": len(config.CHEST_CLASSES),
            "loaded": self.loaded,
        }


registry = ModelRegistry()
