"""
Treina uma ResNet-101 para classificacao de doencas em folhas.

Exemplo rapido em CPU:
    py .\\train_resnet101.py --data-dir data_resnet101 --epochs 3 --batch-size 4 --no-pretrained

Exemplo recomendado com GPU:
    py .\\train_resnet101.py --data-dir data_resnet101 --epochs 30 --batch-size 16 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina ResNet-101 em dataset ImageFolder.")
    parser.add_argument("--data-dir", type=Path, default=Path("data_resnet101"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_resnet101"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "dml", "directml"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument(
        "--unfreeze-layer4",
        action="store_true",
        help="Treina tambem o ultimo bloco convolucional da ResNet.",
    )
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-one-cycle", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=0)
    parser.add_argument(
        "--stop-at",
        type=str,
        default=None,
        help="Horario local para parar limpo, no formato HH:MM ou YYYY-MM-DDTHH:MM.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_directml_device():
    try:
        import torch_directml

        return torch_directml.device()
    except Exception:
        return None


def is_cuda_device(device) -> bool:
    return getattr(device, "type", None) == "cuda"


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
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA nao esta disponivel neste ambiente.")
    return torch.device(choice)


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


def load_datasets(data_dir: Path, img_size: int) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    train_tfms, eval_tfms = build_transforms(img_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tfms)

    if train_ds.class_to_idx != val_ds.class_to_idx or train_ds.class_to_idx != test_ds.class_to_idx:
        raise RuntimeError("Mapeamento de classes diferente entre train/val/test.")

    return train_ds, val_ds, test_ds


def build_model(
    num_classes: int,
    pretrained: bool,
    unfreeze_backbone: bool,
    unfreeze_layer4: bool,
) -> nn.Module:
    weights = models.ResNet101_Weights.DEFAULT if pretrained else None
    model = models.resnet101(weights=weights)

    if not unfreeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, num_classes),
    )
    return model


def compute_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / (len(counts) * counts.clamp_min(1))
    return weights.to(device)


def limited_loader(loader: DataLoader, limit: int) -> Iterable:
    if limit <= 0:
        yield from loader
        return
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= limit:
            break
        yield batch


def parse_stop_at(value: str | None) -> float | None:
    if not value:
        return None

    now = dt.datetime.now()
    try:
        if "T" in value:
            deadline = dt.datetime.fromisoformat(value)
        else:
            hour, minute = value.split(":", maxsplit=1)
            deadline = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
            if deadline <= now:
                deadline += dt.timedelta(days=1)
    except Exception as exc:
        raise ValueError("--stop-at deve estar no formato HH:MM ou YYYY-MM-DDTHH:MM.") from exc

    return deadline.timestamp()


def deadline_reached(deadline_ts: float | None) -> bool:
    return deadline_ts is not None and time.time() >= deadline_ts


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    limit_batches: int = 0,
    deadline_ts: float | None = None,
) -> tuple[float, float, int, bool]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in limited_loader(loader, limit_batches):
            if deadline_reached(deadline_ts):
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            batches += 1

    if batches == 0 or total == 0:
        return 0.0, 0.0, batches, deadline_reached(deadline_ts)
    return running_loss / batches, correct / total, batches, deadline_reached(deadline_ts)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    limit_batches: int = 0,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in limited_loader(loader, limit_batches):
            outputs = model(images.to(device, non_blocking=True))
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_true.extend(labels.tolist())
            y_pred.extend(preds)
    return y_true, y_pred


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for target, pred in zip(y_true, y_pred):
        matrix[target][pred] += 1
    return matrix


def classification_report(matrix: list[list[int]], classes: list[str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for idx, class_name in enumerate(classes):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(len(classes)) if row != idx)
        fn = sum(matrix[idx][col] for col in range(len(classes)) if col != idx)
        support = sum(matrix[idx])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": class_name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
    return rows


def save_csv_matrix(path: Path, matrix: list[list[int]], classes: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *row])


def save_csv_report(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    values: dict[str, object] = {}
    for key, value in vars(args).items():
        values[key] = str(value) if isinstance(value, Path) else value
    return values


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    deadline_ts = parse_stop_at(args.stop_at)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = load_datasets(args.data_dir, args.img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device(device),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device(device),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device(device),
    )

    model = build_model(
        num_classes=len(train_ds.classes),
        pretrained=not args.no_pretrained,
        unfreeze_backbone=args.unfreeze_backbone,
        unfreeze_layer4=args.unfreeze_layer4,
    ).to(device)
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if list(checkpoint.get("classes", [])) != list(train_ds.classes):
            raise RuntimeError("As classes do checkpoint nao batem com o dataset atual.")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint retomado de: {args.resume_checkpoint}")

    class_weights = None if args.no_class_weights else compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=False,
        )

    steps_per_epoch = args.limit_train_batches or len(train_loader)
    scheduler = None
    if not args.no_one_cycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.learning_rate,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
        )

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_count = 0
    best_path = args.output_dir / "best_resnet101.pt"

    print(f"Device: {device}")
    print(f"Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"Imagens: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    if deadline_ts is not None:
        print(f"Parada limpa agendada para: {dt.datetime.fromtimestamp(deadline_ts).isoformat(timespec='seconds')}")

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if deadline_reached(deadline_ts):
            print("Horario limite atingido antes de iniciar nova epoca.")
            break

        train_loss, train_acc, train_batches, stopped_by_deadline = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            limit_batches=args.limit_train_batches,
            deadline_ts=deadline_ts,
        )
        if train_batches == 0:
            print("Horario limite atingido antes de processar lotes de treino.")
            break

        val_loss, val_acc, val_batches, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            limit_batches=args.limit_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "train_batches": train_batches,
            "val_batches": val_batches,
            "partial_epoch": stopped_by_deadline,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"train_batches={train_batches}"
        )

        improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "model_name": "resnet101",
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "classes": train_ds.classes,
                    "img_size": args.img_size,
                    "pretrained": not args.no_pretrained,
                    "args": serializable_args(args),
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                },
                best_path,
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping na epoca {epoch}. Melhor epoca: {best_epoch}.")
                break

        if stopped_by_deadline:
            print("Horario limite atingido durante o treino. Encerrando apos validacao e checkpoint.")
            break

    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not best_path.exists():
        torch.save(
            {
                "model_name": "resnet101",
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "classes": train_ds.classes,
                "img_size": args.img_size,
                "pretrained": not args.no_pretrained,
                "args": serializable_args(args),
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
            },
            best_path,
        )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, _, _ = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        limit_batches=args.limit_test_batches,
    )
    y_true, y_pred = collect_predictions(
        model,
        test_loader,
        device,
        limit_batches=args.limit_test_batches,
    )
    matrix = confusion_matrix(y_true, y_pred, len(train_ds.classes))
    report = classification_report(matrix, train_ds.classes)

    save_csv_matrix(args.output_dir / "confusion_matrix_test.csv", matrix, train_ds.classes)
    save_csv_report(args.output_dir / "classification_report_test.csv", report)

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "elapsed_seconds": round(time.time() - started, 2),
        "classes": train_ds.classes,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Melhor checkpoint salvo em: {best_path}")


if __name__ == "__main__":
    main()
