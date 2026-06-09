import os
import glob
import argparse
import csv
import numpy as np
from PIL import Image
import cv2
from insightface.app import FaceAnalysis


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


def pil_to_bgr(pil_img):
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def get_embedding(app, pil_img):
    img_bgr = pil_to_bgr(pil_img)
    faces = app.get(img_bgr)

    if len(faces) == 0:
        return None

    # Use largest detected face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )

    return face.normed_embedding


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", required=True, help="Directory containing eval_mv_*.jpg")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--cols", type=int, default=13)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--front_col", type=int, default=6)
    parser.add_argument("--ctx_id", type=int, default=-1, help="-1 = CPU, 0 = GPU")
    args = parser.parse_args()

    print(f"Using ArcFace ctx_id={args.ctx_id} (-1 means CPU)")

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=args.ctx_id, det_size=(640, 640))

    grid_paths = sorted(glob.glob(os.path.join(args.eval_dir, "eval_mv_*.jpg")))
    if not grid_paths:
        raise FileNotFoundError(f"No eval_mv_*.jpg found in {args.eval_dir}")

    rows_out = []
    similarities = []
    failed = 0

    for grid_path in grid_paths:
        name = os.path.basename(grid_path)

        # row 0 = stylized, row 1 = original
        stylized = crop_cell(grid_path, args.cols, args.rows, row=0, col=args.front_col)
        original = crop_cell(grid_path, args.cols, args.rows, row=1, col=args.front_col)

        emb_orig = get_embedding(app, original)
        emb_styl = get_embedding(app, stylized)

        if emb_orig is None or emb_styl is None:
            sim = ""
            status = "face_not_detected"
            failed += 1
        else:
            sim = cosine_similarity(emb_orig, emb_styl)
            similarities.append(sim)
            status = "ok"

        rows_out.append([name, sim, status])
        print(f"{name} | similarity={sim} | status={status}")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "arcface_id_similarity", "status"])
        writer.writerows(rows_out)

        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            writer.writerow(["AVERAGE", avg_sim, f"valid={len(similarities)}, failed={failed}"])
            print(f"AVERAGE | similarity={avg_sim:.4f} | valid={len(similarities)} | failed={failed}")
        else:
            print("No valid face pairs were detected.")

    print(f"Saved results to {args.out_csv}")


if __name__ == "__main__":
    main()