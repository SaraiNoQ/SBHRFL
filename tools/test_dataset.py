import os
from PIL import Image
root = "data/VLCS"
for split in ["train", "test"]:
    split_dir = os.path.join(root, split)
    for dp, _, fs in os.walk(split_dir):
        for fn in fs:
            if fn.lower().endswith((".jpg",".jpeg",".png",".bmp",".gif")):
                p = os.path.join(dp, fn)
                try:
                    with Image.open(p) as im:
                        im.convert("RGB").load()  # 真正触发截断/损坏错误
                except Exception as e:
                    print(f"BAD: {p} -> {e}")