from PIL import Image, ImageDraw
import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--pixar_dir", required=True, help="Directory with Pixar eval_mv_*.jpg files")
parser.add_argument("--cyber_dir", required=True, help="Directory with Cyberpunk eval_mv_*.jpg files")
parser.add_argument("--oil_dir", required=True, help="Directory with Oil Paint eval_mv_*.jpg files")
parser.add_argument("--out", required=True, help="Output GIF path")
parser.add_argument("--cols", type=int, default=13, help="Number of view columns in each grid")
parser.add_argument("--rows", type=int, default=2, help="Number of rows in each grid")
parser.add_argument("--fps", type=int, default=6)
parser.add_argument("--padding", type=int, default=10)
parser.add_argument("--label_height", type=int, default=24)
args = parser.parse_args()

# Collect matching files
pixar_files = sorted(glob.glob(os.path.join(args.pixar_dir, "eval_mv_*.jpg")))
cyber_files = sorted(glob.glob(os.path.join(args.cyber_dir, "eval_mv_*.jpg")))
oil_files = sorted(glob.glob(os.path.join(args.oil_dir, "eval_mv_*.jpg")))

pixar_map = {os.path.basename(f): f for f in pixar_files}
cyber_map = {os.path.basename(f): f for f in cyber_files}
oil_map = {os.path.basename(f): f for f in oil_files}

common_names = sorted(set(pixar_map.keys()) & set(cyber_map.keys()) & set(oil_map.keys()))
if not common_names:
    raise ValueError("No matching eval_mv_*.jpg files found between Pixar, Cyberpunk, and Oil Paint directories.")

# Open first image to infer cell size
sample_img = Image.open(pixar_map[common_names[0]]).convert("RGB")
w, h = sample_img.size
cell_w = w // args.cols
cell_h = h // args.rows

# Load all grids
pixar_imgs = {name: Image.open(pixar_map[name]).convert("RGB") for name in common_names}
cyber_imgs = {name: Image.open(cyber_map[name]).convert("RGB") for name in common_names}
oil_imgs = {name: Image.open(oil_map[name]).convert("RGB") for name in common_names}

n_ids = len(common_names)
panel_w = cell_w
panel_h = args.label_height + cell_h * 4

canvas_w = n_ids * panel_w + (n_ids + 1) * args.padding
canvas_h = panel_h + 2 * args.padding

frames = []

for angle_idx in range(args.cols):
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for j, name in enumerate(common_names):
        left = angle_idx * cell_w
        right = left + cell_w

        pixar_img = pixar_imgs[name]
        cyber_img = cyber_imgs[name]
        oil_img = oil_imgs[name]

        # From Pixar grid:
        # row 0 = stylized Pixar
        # row 1 = original
        orig = pixar_img.crop((left, 1 * cell_h, right, 2 * cell_h))
        pixar = pixar_img.crop((left, 0 * cell_h, right, 1 * cell_h))

        # From Cyberpunk grid:
        # row 0 = stylized Cyberpunk
        cyber = cyber_img.crop((left, 0 * cell_h, right, 1 * cell_h))

        # From Oil Paint grid:
        # row 0 = stylized Oil Paint
        oil = oil_img.crop((left, 0 * cell_h, right, 1 * cell_h))

        x = args.padding + j * (panel_w + args.padding)
        y = args.padding

        # Optional identity label
        label = name.replace(".jpg", "")
        draw.text((x + 5, y + 4), label, fill=(0, 0, 0))

        canvas.paste(orig, (x, y + args.label_height + 0 * cell_h))
        canvas.paste(pixar, (x, y + args.label_height + 1 * cell_h))
        canvas.paste(cyber, (x, y + args.label_height + 2 * cell_h))
        canvas.paste(oil, (x, y + args.label_height + 3 * cell_h))

    # Optional row labels on the left
    label_x = 5
    row0_y = args.padding + args.label_height + cell_h // 2 - 8
    row1_y = args.padding + args.label_height + cell_h + cell_h // 2 - 8
    row2_y = args.padding + args.label_height + 2 * cell_h + cell_h // 2 - 8
    row3_y = args.padding + args.label_height + 3 * cell_h + cell_h // 2 - 8
    draw.text((label_x, row0_y), "Original", fill=(0, 0, 0))
    draw.text((label_x, row1_y), "Pixar", fill=(0, 0, 0))
    draw.text((label_x, row2_y), "Cyber", fill=(0, 0, 0))
    draw.text((label_x, row3_y), "Oil", fill=(0, 0, 0))

    frames.append(canvas)

# Smooth loop
frames.append(frames[0])

duration = int(1000 / args.fps)

frames[0].save(
    args.out,
    save_all=True,
    append_images=frames[1:],
    duration=duration,
    loop=0,
)

print(f"Saved combined GIF to {args.out}")