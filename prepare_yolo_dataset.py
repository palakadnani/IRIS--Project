import os
import random
import shutil

# ================= CONFIG =================
SOURCE_DIR = "dataset_chunks"     # folder containing chunk_0, chunk_1, ...
OUTPUT_DIR = "dataset"            # final YOLO dataset folder
TRAIN_RATIO = 0.8                 # 80% train, 20% val
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
# ==========================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def collect_labeled_pairs(source_dir):
    pairs = []

    for root, _, files in os.walk(source_dir):
        for file in files:
            if not is_image_file(file):
                continue

            img_path = os.path.join(root, file)
            base, _ = os.path.splitext(file)
            label_path = os.path.join(root, base + ".txt")

            if os.path.exists(label_path):
                pairs.append((img_path, label_path))
            else:
                print(f"[WARN] No label found for image: {img_path}")

    return pairs


def split_pairs(pairs, train_ratio, seed):
    random.seed(seed)
    random.shuffle(pairs)

    split_idx = int(len(pairs) * train_ratio)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    return train_pairs, val_pairs


def copy_pairs(pairs, image_out_dir, label_out_dir):
    ensure_dir(image_out_dir)
    ensure_dir(label_out_dir)

    for img_path, label_path in pairs:
        img_name = os.path.basename(img_path)
        label_name = os.path.basename(label_path)

        shutil.copy2(img_path, os.path.join(image_out_dir, img_name))
        shutil.copy2(label_path, os.path.join(label_out_dir, label_name))


def main():
    ensure_dir(OUTPUT_DIR)

    train_img_dir = os.path.join(OUTPUT_DIR, "images", "train")
    val_img_dir = os.path.join(OUTPUT_DIR, "images", "val")
    train_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "train")
    val_lbl_dir = os.path.join(OUTPUT_DIR, "labels", "val")

    pairs = collect_labeled_pairs(SOURCE_DIR)

    if not pairs:
        print("[ERROR] No labeled image-label pairs found.")
        return

    print(f"[INFO] Found {len(pairs)} labeled pairs.")

    train_pairs, val_pairs = split_pairs(pairs, TRAIN_RATIO, SEED)

    print(f"[INFO] Train pairs: {len(train_pairs)}")
    print(f"[INFO] Val pairs:   {len(val_pairs)}")

    copy_pairs(train_pairs, train_img_dir, train_lbl_dir)
    copy_pairs(val_pairs, val_img_dir, val_lbl_dir)

    print("[DONE] Dataset prepared successfully.")
    print(f"[DONE] Output folder: {OUTPUT_DIR}")


main()