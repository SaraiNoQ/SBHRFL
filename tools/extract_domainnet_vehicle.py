import os
import zipfile
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ------------------------------
# 配置
# ------------------------------

# 六个子域的 zip 文件名称（你提供的）
DOMAINS = [
    "clipart.zip",
    "infograph.zip",
    "painting.zip",
    "quickdraw.zip",
    "real.zip",
    "sketch.zip"
]

# 道路车辆类别（12类）
VEHICLE_CLASSES = [
    "ambulance",
    "bicycle",
    "bus",
    "car",
    "firetruck",
    "motorbike",
    "police_car",
    "school_bus",
    "tractor",
    "truck",
    "pickup_truck",
    "van",
]

# Train / Test 划分比例
TRAIN_RATIO = 0.8

# ------------------------------
# 解压工具
# ------------------------------

def unzip_file(zip_path, extract_to):
    """解压 zip 文件"""
    print(f"📦 解压中: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)


# ------------------------------
# 抽取车辆类
# ------------------------------

def extract_vehicle_classes(domain_path, output_root):
    """
    从某个 domain 中抽取 12 类车辆图片
    并暂存到 output_root/temp/domain/classname/ 中
    """
    domain_name = os.path.basename(domain_path)
    temp_output = os.path.join(output_root, "temp", domain_name)
    os.makedirs(temp_output, exist_ok=True)

    for cls in VEHICLE_CLASSES:
        src_cls_dir = os.path.join(domain_path, cls)
        dst_cls_dir = os.path.join(temp_output, cls)

        if not os.path.exists(src_cls_dir):
            continue

        os.makedirs(dst_cls_dir, exist_ok=True)

        # 拷贝所有图片
        for img_name in os.listdir(src_cls_dir):
            src_img = os.path.join(src_cls_dir, img_name)
            dst_img = os.path.join(dst_cls_dir, img_name)
            shutil.copy(src_img, dst_img)

    print(f"✔ 已抽取: {domain_name}")


# ------------------------------
# 划分 Train / Test
# ------------------------------

def split_train_test(temp_root, final_root, train_ratio=0.8):
    """
    将全部 domain 合并后，按类划分 train/test
    """
    train_root = os.path.join(final_root, "train")
    test_root = os.path.join(final_root, "test")
    os.makedirs(train_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    print("📂 开始划分 Train / Test ...")

    for cls in VEHICLE_CLASSES:
        # 全部临时目录中该类的图片路径
        cls_dirs = list(Path(temp_root).glob(f"*/{cls}"))
        all_images = []

        for d in cls_dirs:
            for img_path in d.glob("*.*"):
                all_images.append(str(img_path))

        print(f"类别 {cls}: 总计 {len(all_images)} 张")

        # 随机划分
        random.shuffle(all_images)
        split_idx = int(len(all_images) * train_ratio)
        train_imgs = all_images[:split_idx]
        test_imgs = all_images[split_idx:]

        # 创建目录
        os.makedirs(os.path.join(train_root, cls), exist_ok=True)
        os.makedirs(os.path.join(test_root, cls), exist_ok=True)

        # 拷贝 train
        for img in tqdm(train_imgs, desc=f"Train-{cls}"):
            shutil.copy(img, os.path.join(train_root, cls))

        # 拷贝 test
        for img in tqdm(test_imgs, desc=f"Test-{cls}"):
            shutil.copy(img, os.path.join(test_root, cls))

    print("🎉 Train/Test 划分完成！")


# ------------------------------
# 主函数
# ------------------------------

def main(zip_dir, output_root):
    """
    zip_dir:   DomainNet 的 zip 文件所在目录
    output_root:  输出根目录，例如 ./domainnet_vehicle_dataset
    """
    print("🚀 开始处理 DomainNet 六个子域...")

    unzipped_dir = os.path.join(output_root, "unzipped")
    os.makedirs(unzipped_dir, exist_ok=True)

    # 1. 解压所有 zip
    for zip_name in DOMAINS:
        zip_path = os.path.join(zip_dir, zip_name)
        unzip_file(zip_path, unzipped_dir)

    # 2. 抽取车辆类文件
    for domain in DOMAINS:
        domain_name = domain.replace(".zip", "")
        domain_path = os.path.join(unzipped_dir, domain_name)
        extract_vehicle_classes(domain_path, output_root)

    # 3. 划分 Train / Test (ImageFolder)
    temp_root = os.path.join(output_root, "temp")
    final_root = os.path.join(output_root, "imagefolder_vehicle")
    split_train_test(temp_root, final_root, TRAIN_RATIO)

    print("🎯 全部完成！最终数据集位于：", final_root)


# ------------------------------

if __name__ == "__main__":
    # 修改成你的 zip 文件所在目录和输出目录
    ZIP_DIR = "./zips"
    OUTPUT_ROOT = "./domainnet_vehicle_output"

    main(ZIP_DIR, OUTPUT_ROOT)
