from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--grid", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--cols", type=int, default=13)
parser.add_argument("--rows", type=int, default=2)
parser.add_argument("--row", type=int, default=0, help="0 = stylized row, 1 = original row")
parser.add_argument("--fps", type=int, default=6)
args = parser.parse_args()

img = Image.open(args.grid).convert("RGB")
w, h = img.size

cell_w = w // args.cols
cell_h = h // args.rows

frames = []
for i in range(args.cols):
    left = i * cell_w
    upper = args.row * cell_h
    right = left + cell_w
    lower = upper + cell_h
    frame = img.crop((left, upper, right, lower))
    frames.append(frame)

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

print(f"Saved GIF to {args.out}")

"""

python3 make_video_from_grid.py \
  --grid work_dirs/royal_cyberpunk_sds_eval/eval_mv_0000.jpg \
  --out work_dirs/royal_cyberpunk_sds_eval/royal_cyberpunk_stylized_360.gif \
  --cols 13 \
  --rows 2 \
  --row 0 \
  --fps 6

"""