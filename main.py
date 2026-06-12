from __future__ import annotations

import base64
import os
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.responses import JSONResponse
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_MODEL_PATHS = [
    Path("outputs_resnet101_full/best_resnet101.pt"),
    Path("outputs_resnet101_until_2100/best_resnet101.pt"),
    Path("outputs_resnet101/best_resnet101.pt"),
    Path("best_resnet101.pt"),
]
DEFAULT_ALLOWED_SITE_URL = "https://rafael-casarin.github.io/testeLab/"

CLASS_DISPLAY_NAMES = {
    "bacterial_blight": "Mancha bacteriana",
    "brown_spot": "Mancha marrom",
    "cercospora_leaf_blight": "Crestamento foliar por Cercospora",
    "crestamento": "Crestamento",
    "downey_mildew": "Mildio",
    "ferrugem": "Ferrugem",
    "frogeye": "Olho-de-ra",
    "healthy": "Folha saudavel",
    "mosaic_virus": "Virus do mosaico",
    "potassium_deficiency": "Deficiencia de potassio",
    "powdery_mildew": "Oidio",
    "septoria": "Septoriose",
    "southern_blight": "Murcha de Sclerotium",
    "soybean_rust": "Ferrugem asiatica",
    "sudden_death_syndrome": "Sindrome da morte subita",
    "target_spot": "Mancha alvo",
    "yellow_mosaic": "Mosaico amarelo",
}


def normalize_origin(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return value.strip().rstrip("/")


def parse_allowed_origins() -> set[str]:
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS", DEFAULT_ALLOWED_SITE_URL)
    return {
        normalize_origin(origin)
        for origin in raw_origins.split(",")
        if origin.strip()
    }


ALLOWED_ORIGINS = parse_allowed_origins()


app = FastAPI(
    title="API ResNet101 Soja",
    description="Classificacao de doencas em folhas de soja por imagem.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


model: nn.Module | None = None
classes: list[str] = []
img_size = 224
model_source = ""


@app.middleware("http")
async def bloquear_origens(request: Request, call_next):
    origin = request.headers.get("origin")

    if origin and normalize_origin(origin) not in ALLOWED_ORIGINS:
        return JSONResponse(
            status_code=403,
            content={"erro": "Origem nao permitida"},
        )

    return await call_next(request)


def build_model(num_classes: int) -> nn.Module:
    resnet = models.resnet101(weights=None)
    in_features = resnet.fc.in_features
    resnet.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, num_classes),
    )
    return resnet


def resolve_model_path() -> Path:
    explicit_path = os.getenv("MODEL_PATH")
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"MODEL_PATH nao encontrado: {path}")
        return path

    model_url = os.getenv("MODEL_URL")
    if model_url:
        target = Path(os.getenv("MODEL_CACHE_PATH", "/tmp/best_resnet101.pt"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            urllib.request.urlretrieve(model_url, target)
        return target

    for path in DEFAULT_MODEL_PATHS:
        if path.exists():
            return path

    searched = ", ".join(str(path) for path in DEFAULT_MODEL_PATHS)
    raise FileNotFoundError(
        "Nenhum checkpoint encontrado. Configure MODEL_URL no Render ou MODEL_PATH local. "
        f"Caminhos locais procurados: {searched}"
    )


def load_model() -> None:
    global model, classes, img_size, model_source

    path = resolve_model_path()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except AttributeError as exc:
        raise RuntimeError(
            "Falha ao carregar o checkpoint. Verifique se MODEL_URL aponta para o arquivo "
            "ResNet101 deste projeto, por exemplo outputs_resnet101_full/best_resnet101.pt. "
            "Esse erro costuma acontecer quando MODEL_URL aponta para um best.pt de "
            "YOLO/Ultralytics em vez do checkpoint ResNet101."
        ) from exc

    required_keys = {"classes", "model_state_dict"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise RuntimeError(
            "Checkpoint invalido para esta API. Use o arquivo best_resnet101.pt gerado "
            "por train_resnet101.py, nao um modelo YOLO/Ultralytics."
        )

    checkpoint_classes = list(checkpoint["classes"])
    loaded_model = build_model(len(checkpoint_classes))
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    model = loaded_model
    classes = checkpoint_classes
    img_size = int(checkpoint.get("img_size", 224))
    model_source = str(path)


def get_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def open_image(data: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Arquivo enviado nao e uma imagem valida.") from exc


def format_prediction(class_name: str, confidence: float) -> dict[str, Any]:
    return {
        "class": class_name,
        "label": CLASS_DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title()),
        "confidence": round(confidence, 6),
        "is_healthy": class_name == "healthy",
    }


def predict_image(image: Image.Image, top_k: int) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ainda nao carregado.")

    tensor = get_transform()(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0)

    k = min(max(top_k, 1), len(classes))
    scores, indices = torch.topk(probs, k=k)
    predictions = [
        format_prediction(classes[idx], float(score))
        for score, idx in zip(scores.tolist(), indices.tolist())
    ]
    return {
        "result": predictions[0],
        "predictions": predictions,
        "model": {
            "architecture": "resnet101",
            "classes": len(classes),
            "source": model_source,
        },
    }


@app.on_event("startup")
def startup_event() -> None:
    load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "model_source": model_source,
        "classes": len(classes),
    }


@app.get("/classes")
def list_classes() -> dict[str, Any]:
    return {
        "classes": [
            {"class": class_name, "label": CLASS_DISPLAY_NAMES.get(class_name, class_name)}
            for class_name in classes
        ]
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=10),
) -> dict[str, Any]:
    data = await file.read()
    image = open_image(data)
    return predict_image(image, top_k)


@app.post("/predict-base64")
async def predict_base64(
    payload: dict[str, str],
    top_k: int = Query(default=3, ge=1, le=10),
) -> dict[str, Any]:
    raw_image = payload.get("image")
    if not raw_image:
        raise HTTPException(status_code=400, detail="Envie o campo 'image' em base64.")

    if "," in raw_image:
        raw_image = raw_image.split(",", maxsplit=1)[1]

    try:
        data = base64.b64decode(raw_image, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Base64 invalido.") from exc

    image = open_image(data)
    return predict_image(image, top_k)
