import glob
import argparse
import csv
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import numpy as np
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "str"):
    np.str = str

def crop_cell(grid_path, cols=13, rows=2, row=0, col=6):
    img = Image.open(grid_path).convert("RGB")
    w, h = img.size
    cell_w = w // cols
    cell_h = h // rows

    left = col * cell_w
    upper = row * cell_h
    right = left + cell_w
    lower = upper + cell_h

    return img.crop((left, upper, right, lower))


def clip_score(model, processor, image, prompt, device):
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        img_feat = outputs.image_embeds
        text_feat = outputs.text_embeds

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        return float((img_feat @ text_feat.T).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--cols", type=int, default=13)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--front_col", type=int, default=6)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    grid_paths = sorted(glob.glob(os.path.join(args.eval_dir, "eval_mv_*.jpg")))
    if not grid_paths:
        raise FileNotFoundError(f"No eval_mv_*.jpg found in {args.eval_dir}")

    rows_out = []
    before_scores = []
    after_scores = []
    delta_scores = []

    for grid_path in grid_paths:
        name = os.path.basename(grid_path)

        # row 0 = stylized, row 1 = original
        stylized = crop_cell(grid_path, args.cols, args.rows, row=0, col=args.front_col)
        original = crop_cell(grid_path, args.cols, args.rows, row=1, col=args.front_col)

        before = clip_score(model, processor, original, args.prompt, device)
        after = clip_score(model, processor, stylized, args.prompt, device)
        delta = after - before

        before_scores.append(before)
        after_scores.append(after)
        delta_scores.append(delta)

        rows_out.append([name, before, after, delta])
        print(f"{name} | before={before:.4f} | after={after:.4f} | delta={delta:.4f}")

    avg_before = sum(before_scores) / len(before_scores)
    avg_after = sum(after_scores) / len(after_scores)
    avg_delta = sum(delta_scores) / len(delta_scores)

    rows_out.append(["AVERAGE", avg_before, avg_after, avg_delta])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "clip_before_original", "clip_after_stylized", "delta_after_minus_before"])
        writer.writerows(rows_out)

    print(f"Saved results to {args.out_csv}")
    print(f"AVERAGE | before={avg_before:.4f} | after={avg_after:.4f} | delta={avg_delta:.4f}")


if __name__ == "__main__":
    main()