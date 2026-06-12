"""
Prediz a classe de uma imagem usando o checkpoint treinado em train_resnet101.py.

Exemplo:
    py .\\predict_resnet101.py --checkpoint outputs_resnet101\\best_resnet101.pt --image healthy\\healthy_1.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicao com ResNet-101 treinada.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs_resnet101/best_resnet101.pt"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml", "directml"], default="auto")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def get_directml_device():
    try:
        import torch_directml

        return torch_directml.device()
    except Exception:
        return None


def resolve_device(choice: str):
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        directml_device = get_directml_device()
        if directml_device is not None:
            return directml_device
        return torch.device("cpu")
    if choice in {"dml", "directml"}:
        directml_device = get_directml_device()
        if directml_device is None:
            raise RuntimeError("DirectML nao esta disponivel. Instale torch-directml.")
        return directml_device
    return torch.device(choice)


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet101(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, num_classes),
    )
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    classes = checkpoint["classes"]
    img_size = int(checkpoint.get("img_size", 224))
    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()

    top_k = min(args.top_k, len(classes))
    scores, indices = torch.topk(probs, k=top_k)
    for score, idx in zip(scores.tolist(), indices.tolist()):
        print(f"{classes[idx]}: {score:.4f}")


if __name__ == "__main__":
    main()
