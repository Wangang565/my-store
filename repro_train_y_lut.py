import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "repro_yuv420_lut.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproducible Y-channel LUT training for full-range or TV-range absolute Y."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--range", choices=["full", "tv"], required=True, dest="y_range")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--crop-size-lr", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def repo_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def list_images(root):
    root = Path(root)
    files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError("No images found in {}".format(root))
    return files


def load_rgb(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def rgb_to_y_uint8(rgb, y_range):
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if y_range == "tv":
        y = y * (219.0 / 255.0) + 16.0
        y = np.clip(np.round(y), 16, 235)
    else:
        y = np.clip(np.round(y), 0, 255)
    return y.astype(np.uint8)


def normalize_y(y_uint8, y_range):
    y = y_uint8.astype(np.float32)
    if y_range == "tv":
        return (y - 16.0) / 219.0
    return y / 255.0


def denormalize_y(y_norm, y_range):
    if y_range == "tv":
        return np.clip(np.round(y_norm * 219.0 + 16.0), 16, 235).astype(np.uint8)
    return np.clip(np.round(y_norm * 255.0), 0, 255).astype(np.uint8)


def box_downsample_x4(y_hr):
    h, w = y_hr.shape
    h4 = h - (h % 4)
    w4 = w - (w % 4)
    y = y_hr[:h4, :w4].astype(np.float32)
    return np.round(y.reshape(h4 // 4, 4, w4 // 4, 4).mean(axis=(1, 3))).astype(np.uint8)


def crop_hr_for_lr(y_hr, crop_size_lr):
    crop_hr = crop_size_lr * 4
    h, w = y_hr.shape
    h4 = h - (h % 4)
    w4 = w - (w % 4)
    if h4 < crop_hr or w4 < crop_hr:
        raise ValueError("Image is too small for LR crop {}: {}x{}".format(crop_size_lr, w, h))
    max_y = (h4 - crop_hr) // 4
    max_x = (w4 - crop_hr) // 4
    y0 = random.randint(0, max_y) * 4
    x0 = random.randint(0, max_x) * 4
    return y_hr[y0 : y0 + crop_hr, x0 : x0 + crop_hr]


def make_batch(files, batch_size, crop_size_lr, y_range):
    lr_items = []
    hr_items = []
    attempts = 0
    while len(lr_items) < batch_size:
        attempts += 1
        if attempts > batch_size * 20:
            raise RuntimeError("Could not build a batch; check image sizes.")
        path = random.choice(files)
        try:
            y_hr_full = rgb_to_y_uint8(load_rgb(path), y_range)
            y_hr = crop_hr_for_lr(y_hr_full, crop_size_lr)
        except ValueError:
            continue
        y_lr = box_downsample_x4(y_hr)
        lr_items.append(normalize_y(y_lr, y_range))
        hr_items.append(normalize_y(y_hr, y_range))
    lr = torch.from_numpy(np.stack(lr_items)[:, None, :, :].astype(np.float32))
    hr = torch.from_numpy(np.stack(hr_items)[:, None, :, :].astype(np.float32))
    return lr, hr


def strict_flat_mask(lr):
    y1 = lr[:, :, :-1, :-1]
    y2 = lr[:, :, :-1, 1:]
    y3 = lr[:, :, 1:, :-1]
    y4 = lr[:, :, 1:, 1:]
    return (y1 == y2) & (y1 == y3) & (y1 == y4)


def expand_block_mask(mask, upscale):
    return mask.repeat_interleave(upscale, dim=2).repeat_interleave(upscale, dim=3)


def copy_base_from_lr(lr, upscale=4):
    y1 = lr[:, :, :-1, :-1]
    y2 = lr[:, :, :-1, 1:]
    y3 = lr[:, :, 1:, :-1]
    y4 = lr[:, :, 1:, 1:]
    b, c, h, w = y1.shape
    blocks = lr.new_empty((b, c, h, w, upscale, upscale))
    blocks[:, :, :, :, 0:2, 0:2] = y1[..., None, None]
    blocks[:, :, :, :, 0:2, 2:4] = y2[..., None, None]
    blocks[:, :, :, :, 2:4, 0:2] = y3[..., None, None]
    blocks[:, :, :, :, 2:4, 2:4] = y4[..., None, None]
    return blocks.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h * upscale, w * upscale)


class DirectYNet(nn.Module):
    def __init__(self, upscale=4, hidden=64):
        super().__init__()
        self.upscale = upscale
        self.hidden = hidden
        self.conv1 = nn.Conv2d(4, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 1)
        self.conv3 = nn.Conv2d(hidden, hidden, 1)
        self.conv4 = nn.Conv2d(hidden, hidden, 1)
        self.conv5 = nn.Conv2d(hidden, hidden, 1)
        self.conv6 = nn.Conv2d(hidden, upscale * upscale, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.reset_to_copy()

    def reset_to_copy(self):
        for module in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6):
            nn.init.constant_(module.weight, 0.0)
            nn.init.constant_(module.bias, 0.0)

        with torch.no_grad():
            for i in range(4):
                self.conv1.weight[i, i, 0, 0] = 1.0
            for conv in (self.conv2, self.conv3, self.conv4, self.conv5):
                for i in range(4):
                    conv.weight[i, i, 0, 0] = 1.0

            mapping = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]
            for out_ch, in_ch in enumerate(mapping):
                self.conv6.weight[out_ch, in_ch, 0, 0] = 1.0

    def forward(self, x):
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=2, stride=1)
        patches = patches.reshape(b, c * 4, h - 1, w - 1)
        patches = patches.reshape(b * c, 4, h - 1, w - 1)
        x = self.conv1(patches)
        x = self.conv2(F.relu(x))
        x = self.conv3(F.relu(x))
        x = self.conv4(F.relu(x))
        x = self.conv5(F.relu(x))
        x = self.conv6(F.relu(x))
        x = self.pixel_shuffle(x)
        return x.reshape(b, c, self.upscale * (h - 1), self.upscale * (w - 1))


def model_predict_4rot(model, lr, apply_flat_copy=True):
    out0 = model(lr)
    out1 = torch.rot90(model(torch.rot90(lr, 1, dims=(2, 3))), 3, dims=(2, 3))
    out2 = torch.rot90(model(torch.rot90(lr, 2, dims=(2, 3))), 2, dims=(2, 3))
    out3 = torch.rot90(model(torch.rot90(lr, 3, dims=(2, 3))), 1, dims=(2, 3))
    pred = (out0 + out1 + out2 + out3) * 0.25
    if apply_flat_copy:
        flat = strict_flat_mask(lr)
        flat_pix = expand_block_mask(flat, model.upscale)
        base = copy_base_from_lr(lr, model.upscale)
        pred = torch.where(flat_pix, base, pred)
    return pred


def masked_l1(pred, target, lr, upscale):
    flat = strict_flat_mask(lr)
    nonflat = ~expand_block_mask(flat, upscale)
    if int(nonflat.sum().item()) == 0:
        return None, 1.0
    loss = torch.abs(pred - target)[nonflat].mean()
    flat_ratio = float(flat.float().mean().item())
    return loss, flat_ratio


def psnr_uint8(ref, pred):
    diff = ref.astype(np.float32) - pred.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def validate(model, files, y_range, crop_size_lr, device, max_images=20):
    model.eval()
    psnrs = []
    flat_ratios = []
    selected = files[:max_images]
    with torch.no_grad():
        for path in selected:
            y_hr_full = rgb_to_y_uint8(load_rgb(path), y_range)
            h4 = y_hr_full.shape[0] - (y_hr_full.shape[0] % 4)
            w4 = y_hr_full.shape[1] - (y_hr_full.shape[1] % 4)
            if h4 < crop_size_lr * 4 or w4 < crop_size_lr * 4:
                continue
            y_hr = y_hr_full[:h4, :w4]
            y_lr = box_downsample_x4(y_hr)
            lr = torch.from_numpy(normalize_y(y_lr, y_range)[None, None].astype(np.float32)).to(device)
            pred = model_predict_4rot(model, lr, apply_flat_copy=True).clamp(0.0, 1.0)
            out_h, out_w = pred.shape[-2], pred.shape[-1]
            target = y_hr[:out_h, :out_w]
            pred_u8 = denormalize_y(pred[0, 0].cpu().numpy(), y_range)
            psnrs.append(psnr_uint8(target, pred_u8))
            flat_ratios.append(float(strict_flat_mask(lr).float().mean().item()))
    model.train()
    return {
        "val_images": len(psnrs),
        "psnr": float(np.mean(psnrs)) if psnrs else 0.0,
        "flat_ratio": float(np.mean(flat_ratios)) if flat_ratios else 0.0,
    }


def save_checkpoint(path, model, optimizer, epoch, step, best_psnr, config, y_range):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_psnr": best_psnr,
            "config": config,
            "range": y_range,
            "model": "DirectYNet",
            "upscale": model.upscale,
        },
        path,
    )


def append_csv(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def update_checkpoint_manifest(manifest_path, entry):
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"checkpoints": []}
    data["checkpoints"] = [
        item for item in data.get("checkpoints", []) if item.get("range") != entry["range"]
    ]
    data["checkpoints"].append(entry)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, data)


def run_self_test(device):
    model = DirectYNet(upscale=4).to(device)
    x = torch.tensor([[[[10.0, 20.0], [30.0, 40.0]]]], device=device) / 255.0
    expected = torch.tensor(
        [
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [30, 30, 40, 40],
        ],
        device=device,
        dtype=torch.float32,
    ) / 255.0
    out = model(x)[0, 0]
    if not torch.allclose(out, expected, atol=1e-6):
        raise AssertionError("copy initialization failed")
    flat = strict_flat_mask(torch.ones((1, 1, 2, 2), device=device) * 0.5)
    nonflat = strict_flat_mask(x)
    if not bool(flat.item()):
        raise AssertionError("strict flat mask failed on flat input")
    if bool(nonflat.item()):
        raise AssertionError("strict flat mask failed on non-flat input")
    print("self-test ok: copy initialization and strict flat mask")


def main():
    args = parse_args()
    config = load_json(args.config)
    train_cfg = config["train"]
    epochs = args.epochs if args.epochs is not None else int(train_cfg["epochs"])
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else int(train_cfg["steps_per_epoch"])
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg["batch_size"])
    crop_size_lr = args.crop_size_lr if args.crop_size_lr is not None else int(train_cfg["crop_size_lr"])
    seed = int(config["seed"])
    set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.self_test:
        run_self_test(device)
        return

    output_root = repo_path(args.output_root or config.get("output_root", "output/experiments"))
    run_id = args.run_id or datetime.now().strftime("train_%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    ckpt_dir = run_dir / "checkpoints" / args.y_range
    log_csv = run_dir / "train_log_{}.csv".format(args.y_range)
    val_csv = run_dir / "val_metrics_{}.csv".format(args.y_range)

    train_files = list_images(repo_path(train_cfg["hr_dir"]))
    val_files = list_images(repo_path(train_cfg["val_hr_dir"]))
    model = DirectYNet(upscale=int(config["upscale"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]))
    start_epoch = 0
    global_step = 0
    best_psnr = -1.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        best_psnr = float(ckpt.get("best_psnr", -1.0))

    resolved = {
        "run_id": run_id,
        "range": args.y_range,
        "config_source": str(Path(args.config).resolve()),
        "repo_root": str(REPO_ROOT),
        "device": str(device),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "crop_size_lr": crop_size_lr,
        "train_images": len(train_files),
        "val_images": len(val_files),
        "config": config,
    }
    write_json(run_dir / "config_resolved_{}.json".format(args.y_range), resolved)

    manifest = {
        "range": args.y_range,
        "latest": str(ckpt_dir / "latest.pt"),
        "best": str(ckpt_dir / "best.pt"),
        "train_log": str(log_csv),
        "val_metrics": str(val_csv),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

    print("range={} device={} train_images={} val_images={}".format(args.y_range, device, len(train_files), len(val_files)))
    t_start = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = []
        epoch_flat = []
        for step in range(steps_per_epoch):
            lr, hr = make_batch(train_files, batch_size, crop_size_lr, args.y_range)
            lr = lr.to(device)
            hr = hr.to(device)
            target = hr[:, :, : (lr.shape[-2] - 1) * model.upscale, : (lr.shape[-1] - 1) * model.upscale]
            pred = model_predict_4rot(model, lr, apply_flat_copy=False).clamp(0.0, 1.0)
            loss, flat_ratio = masked_l1(pred, target, lr, model.upscale)
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss.append(float(loss.item()))
            epoch_flat.append(flat_ratio)
            if global_step == 1 or global_step % 100 == 0:
                print(
                    "epoch={}/{} step={}/{} global_step={} loss={:.6f} flat={:.4f}".format(
                        epoch + 1,
                        epochs,
                        step + 1,
                        steps_per_epoch,
                        global_step,
                        float(loss.item()),
                        flat_ratio,
                    )
                )

        val = validate(model, val_files, args.y_range, crop_size_lr, device)
        avg_loss = float(np.mean(epoch_loss)) if epoch_loss else 0.0
        avg_flat = float(np.mean(epoch_flat)) if epoch_flat else 0.0
        row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "loss_l1_nonflat": "{:.8f}".format(avg_loss),
            "flat_ratio_train": "{:.8f}".format(avg_flat),
            "val_psnr": "{:.6f}".format(val["psnr"]),
            "val_flat_ratio": "{:.8f}".format(val["flat_ratio"]),
            "elapsed_sec": "{:.2f}".format(time.time() - t_start),
        }
        append_csv(
            log_csv,
            ["epoch", "global_step", "loss_l1_nonflat", "flat_ratio_train", "val_psnr", "val_flat_ratio", "elapsed_sec"],
            row,
        )
        append_csv(
            val_csv,
            ["epoch", "global_step", "val_images", "val_psnr", "val_flat_ratio"],
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "val_images": val["val_images"],
                "val_psnr": "{:.6f}".format(val["psnr"]),
                "val_flat_ratio": "{:.8f}".format(val["flat_ratio"]),
            },
        )

        latest_path = ckpt_dir / "latest.pt"
        save_checkpoint(latest_path, model, optimizer, epoch + 1, global_step, best_psnr, config, args.y_range)
        if val["psnr"] >= best_psnr:
            best_psnr = val["psnr"]
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, epoch + 1, global_step, best_psnr, config, args.y_range)
        manifest.update(
            {
                "best_psnr": best_psnr,
                "last_epoch": epoch + 1,
                "last_step": global_step,
                "latest_sha256": sha256_file(latest_path),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        best_path = ckpt_dir / "best.pt"
        if best_path.exists():
            manifest["best_sha256"] = sha256_file(best_path)
        write_json(run_dir / "checkpoint_manifest_{}.json".format(args.y_range), manifest)
        update_checkpoint_manifest(run_dir / "checkpoint_manifest.json", manifest)
        print("epoch={} avg_loss={:.6f} val_psnr={:.4f} best={:.4f}".format(epoch + 1, avg_loss, val["psnr"], best_psnr))

    print("training complete:", run_dir)


if __name__ == "__main__":
    main()
