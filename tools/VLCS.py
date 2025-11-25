import os
import zipfile
import shutil
import random

# ========== 配置部分 ==========
ZIP_PATH = "./archive.zip"   # 你的源 zip 文件路径
EXTRACT_DIR = "./VLCS_raw"  # 解压后的临时目录
DST_ROOT = "./VLCS_imagefolder"  # 输出的 imagefolder 目录

DOMAINS = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
CLASSES = ["bird", "car", "chair", "dog", "person"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TRAIN_RATIO = 0.8
random.seed(42)


# ================== 解压 ==================
def unzip_dataset():
    if not os.path.exists(EXTRACT_DIR):
        print("正在解压 ZIP 文件...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as z:
            z.extractall(EXTRACT_DIR)
    else:
        print("已存在解压目录，跳过解压。")


# ================== 收集所有图像路径 ==================
def collect_images():
    data = {cls: [] for cls in CLASSES}
    vlcs_root = os.path.join(EXTRACT_DIR, "VLCS")  # 解压后内部通常有 VLCS/ 目录

    for domain in DOMAINS:
        for cls in CLASSES:
            cls_dir = os.path.join(vlcs_root, domain, cls)
            if not os.path.isdir(cls_dir):
                print(f"[WARN] 缺失：{cls_dir}")
                continue

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(IMG_EXTS):
                    src = os.path.join(cls_dir, fname)
                    data[cls].append((src, domain))

    return data


# ================== 划分并复制到 imagefolder ==================
def split_and_copy(data):
    for cls, items in data.items():
        print(f"类别 {cls}：共 {len(items)} 张")

        random.shuffle(items)
        n_train = int(len(items) * TRAIN_RATIO)
        train_items = items[:n_train]
        test_items = items[n_train:]

        for split, imgs in [("train", train_items), ("test", test_items)]:
            out_dir = os.path.join(DST_ROOT, split, cls)
            os.makedirs(out_dir, exist_ok=True)

            for idx, (src, domain) in enumerate(imgs):
                base = os.path.basename(src)
                dst_name = f"{domain}_{base}"
                dst_path = os.path.join(out_dir, dst_name)

                if os.path.exists(dst_path):
                    name, ext = os.path.splitext(dst_name)
                    dst_path = os.path.join(out_dir, f"{name}_{idx}{ext}")

                shutil.copy2(src, dst_path)

            print(f"  -> {split}: {len(imgs)} 张已保存")


# ================== 主程序 ==================
def main():
    unzip_dataset()
    data = collect_images()
    split_and_copy(data)
    print("\n🎉 完成！你现在可以使用 ImageFolder 加载数据了。")


if __name__ == "__main__":
    main()
