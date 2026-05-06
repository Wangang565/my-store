import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "configs" / "repro_yuv420_lut.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Export absolute Y LUT from repro DirectYNet checkpoint.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--range", choices=["full", "tv"], dest="y_range")
    parser.add_argument("--output")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
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


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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


def sampling_points(y_range, sampling_interval):
    step = 2 ** sampling_interval
    if y_range == "tv":
        points = list(range(16, 225, step))
        if points[-1] != 235:
            points.append(235)
        return np.asarray(points, dtype=np.uint8)
    pts = list(range(0, 256, step))
    if pts[-1] != 255:
        pts.append(255)
    else:
        pts[-1] = 255
    return np.asarray(pts, dtype=np.uint8)


def normalize_values(vals, y_range):
    vals = vals.astype(np.float32)
    if y_range == "tv":
        return np.clip((vals - 16.0) / 219.0, 0.0, 1.0)
    return vals / 255.0


def denormalize_tensor(tensor, y_range):
    if y_range == "tv":
        return (tensor.clamp(0.0, 1.0) * 219.0 + 16.0).round().clamp(16, 235)
    return (tensor.clamp(0.0, 1.0) * 255.0).round().clamp(0, 255)


def build_grid(points):
    meshes = np.meshgrid(points, points, points, points, indexing="ij")
    combos = np.stack([m.reshape(-1) for m in meshes], axis=1)
    return combos.astype(np.uint8)


def update_lut_manifest(manifest_path, entry):
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"luts": []}
    data["luts"] = [
        item
        for item in data.get("luts", [])
        if not (item.get("range") == entry["range"] and item.get("output") == entry["output"])
    ]
    data["luts"].append(entry)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(manifest_path, data)


def self_test_sampling():
    full = sampling_points("full", 4).tolist()
    tv = sampling_points("tv", 4).tolist()
    expected_full = [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]
    expected_tv = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 235]
    assert full == expected_full
    assert tv == expected_tv
    print("self-test ok: sampling points")


def main():
    args = parse_args()
    if args.self_test:
        self_test_sampling()
        return
    if args.checkpoint is None or args.y_range is None or args.output is None:
        raise ValueError("--checkpoint, --range, and --output are required unless --self-test is used")

    config = load_json(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_range = ckpt.get("range")
    if ckpt_range is not None and ckpt_range != args.y_range:
        raise ValueError("Checkpoint range {} does not match requested export range {}".format(ckpt_range, args.y_range))
    upscale = int(ckpt.get("upscale", config.get("upscale", 4)))
    model = DirectYNet(upscale=upscale).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    points = sampling_points(args.y_range, int(config.get("sampling_interval", 4)))
    combos = build_grid(points)
    outputs = []
    print("range={} device={} points={} total={}".format(args.y_range, device, points.tolist(), len(combos)))
    with torch.no_grad():
        for start in range(0, len(combos), args.batch_size):
            batch = combos[start : start + args.batch_size]
            norm = normalize_values(batch, args.y_range)
            x = torch.from_numpy(norm.reshape(-1, 1, 2, 2).astype(np.float32)).to(device)
            y = model(x)
            y_u8 = denormalize_tensor(y, args.y_range).to(torch.uint8).cpu().numpy()
            outputs.append(y_u8)
            if start == 0 or (start // args.batch_size) % 20 == 0:
                print("exported {}/{}".format(min(start + args.batch_size, len(combos)), len(combos)))

    lut = np.concatenate(outputs, axis=0).astype(np.uint8)
    np.save(output_path, lut)
    entry = {
        "range": args.y_range,
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "output_sha256": sha256_file(output_path),
        "shape": list(lut.shape),
        "reshape_for_inference": [int(lut.shape[0]), upscale * upscale],
        "dtype": str(lut.dtype),
        "min": int(lut.min()),
        "max": int(lut.max()),
        "sampling_interval": int(config.get("sampling_interval", 4)),
        "sampling_points": points.astype(int).tolist(),
        "lut_layout": "compact_tv_15_points" if args.y_range == "tv" else "full_17_points",
        "effective_input_points": points.astype(int).tolist(),
        "upscale": upscale,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "config_source": str(Path(args.config).resolve()),
    }
    write_json(output_path.with_name(output_path.stem + "_manifest.json"), entry)
    update_lut_manifest(output_path.parent / "lut_manifest.json", entry)
    print("saved:", output_path)
    print("shape={} dtype={} range={}..{}".format(lut.shape, lut.dtype, int(lut.min()), int(lut.max())))


if __name__ == "__main__":
    main()
