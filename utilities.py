import shutil
import random
import concurrent.futures
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import os


# ── Split ────────────────────────────────────────────────────────────────────

def split_dataset(
    src_root,
    train_ratio=0.80,
    val_ratio=0.10,
    seed=42,
    workers=8
):
    """
    Merges existing train/val/test (or train/test) folders and re-splits
    into train/val/test in-place, preserving emotion subfolder structure.
    """
    random.seed(seed)
    src_root = Path(src_root).resolve()
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    if not (src_root / 'train').exists():
        raise FileNotFoundError(f"Missing folder: {src_root / 'train'}")

    existing_splits = [
        s for s in ['train', 'val', 'test']
        if (src_root / s).exists()
    ]

    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    emotion_classes = sorted([
        d.name for d in (src_root / 'train').iterdir() if d.is_dir()
    ])

    if not emotion_classes:
        raise ValueError(f"No emotion subfolders found in: {src_root / 'train'}")

    print(f"Dataset:        {src_root}")
    print(f"Classes:        {emotion_classes}")
    print(f"Existing splits: {existing_splits}")
    print(f"Split:          train={train_ratio:.0%} | val={val_ratio:.0%} | test={test_ratio:.0%} | seed={seed}\n")

    stats = defaultdict(lambda: defaultdict(int))

    all_class_images = {}
    for cls in emotion_classes:
        images = []
        for split in existing_splits:
            cls_dir = src_root / split / cls
            if cls_dir.exists():
                images += [
                    f for f in cls_dir.iterdir()
                    if f.suffix.lower() in extensions
                ]
        all_class_images[cls] = images

    tmp_root = src_root / '_tmp_split'
    tmp_root.mkdir(exist_ok=True)

    move_tasks = []
    for cls, images in all_class_images.items():
        tmp_cls = tmp_root / cls
        tmp_cls.mkdir(exist_ok=True)
        for img in images:
            move_tasks.append((img, tmp_cls / img.name))

    def _move_file(args):
        src, dst = args
        shutil.move(str(src), dst)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(
            executor.map(_move_file, move_tasks),
            total=len(move_tasks),
            desc="Collecting"
        ))

    for split in existing_splits:
        shutil.rmtree(src_root / split, ignore_errors=True)

    redistribute_tasks = []
    for cls in emotion_classes:
        images = list((tmp_root / cls).iterdir())
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val   = int(n_total * val_ratio)

        splits = {
            'train': images[:n_train],
            'val':   images[n_train:n_train + n_val],
            'test':  images[n_train + n_val:]
        }

        for split_name, imgs in splits.items():
            dst_cls = src_root / split_name / cls
            dst_cls.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                redistribute_tasks.append((img, dst_cls / img.name))
            stats[split_name][cls] = len(imgs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(
            executor.map(_move_file, redistribute_tasks),
            total=len(redistribute_tasks),
            desc="Distributing"
        ))

    shutil.rmtree(tmp_root)

    _print_split_summary(stats, emotion_classes)
    print(f"\nDone. Dataset modified in-place: {src_root}")


# ── Resize ───────────────────────────────────────────────────────────────────

def _resize_image(args):
    src_path, dst_path, target_size = args
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            img = img.convert('RGB')
            img = img.resize(target_size, Image.LANCZOS)
            img.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception as e:
        print(f"Error: {src_path} -> {e}")
        return False


def resize_dataset_inplace(src_root, target_size=(224, 224), workers=8):
    """
    Resizes all images in src_root in-place, overwriting originals.
    Converts all images to JPEG.
    """
    src_root = Path(src_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory not found: {src_root}")

    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = [
        p for p in src_root.rglob('*')
        if p.suffix.lower() in extensions
    ]

    if not all_images:
        raise ValueError(f"No images found in: {src_root}")

    print(f"Source:      {src_root}")
    print(f"Target size: {target_size[0]}x{target_size[1]} (in-place)")
    print()
    for split in ['train', 'val', 'test']:
        count = sum(1 for p in all_images if split in p.parts)
        if count:
            print(f"  {split}: {count} images")
    print(f"  TOTAL: {len(all_images)}\n")

    tasks = [
        (src, src.with_suffix('.jpg'), target_size)
        for src in all_images
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(_resize_image, tasks),
            total=len(tasks),
            desc="Resizing"
        ))

    for src, dst, _ in tasks:
        if src != dst and src.exists():
            src.unlink()

    ok = sum(results)
    failed = len(tasks) - ok
    print(f"\nDone: {ok}/{len(tasks)} images resized successfully")
    if failed:
        print(f"Failed: {failed} images — check errors above")


# ── List Files ───────────────────────────────────────────────────────────────────

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{sub_indent}{f}')


# ── Helpers ──────────────────────────────────────────────────────────────────

def _print_split_summary(stats, emotion_classes):
    print(f"{'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 47)
    for cls in emotion_classes:
        tr = stats['train'][cls]
        vl = stats['val'][cls]
        te = stats['test'][cls]
        print(f"{cls:<15} {tr:>8} {vl:>8} {te:>8} {tr+vl+te:>8}")
    print("-" * 47)
    total_tr = sum(stats['train'].values())
    total_vl = sum(stats['val'].values())
    total_te = sum(stats['test'].values())
    total    = total_tr + total_vl + total_te
    print(f"{'TOTAL':<15} {total_tr:>8} {total_vl:>8} {total_te:>8} {total:>8}")
    print(f"{'%':<15} {total_tr/total:>8.1%} {total_vl/total:>8.1%} {total_te/total:>8.1%}")