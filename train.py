import os, shutil, random, yaml, cv2, colorsys, argparse, logging, sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "labal"
TRAIN_DIR = BASE_DIR / "dataset/train"
VAL_DIR = BASE_DIR / "dataset/val"
DATA_YAML = BASE_DIR / "data.yaml"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = "yolo26n.pt"
IMG_SIZE = 640
SPLIT_RATIO = 0.8

parser = argparse.ArgumentParser(description="YOLO Training Pipeline")
parser.add_argument("--epochs", type=int, default=None, help="Training epochs (auto if not set)")
parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help=f"Image size (default: {IMG_SIZE})")
parser.add_argument("--device", type=str, default="auto", help="Device: cpu, 0, or auto")
parser.add_argument("--no-label", action="store_true", help="Skip auto-labeling")
parser.add_argument("--no-augment", action="store_true", help="Skip augmentation")
parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
parser.add_argument("--model", type=str, default=MODEL_FILE, help="Base model file")
parser.add_argument("--name", type=str, default=None, help="Experiment name")
args = parser.parse_args()

logo = """
  YOLO Training v2
  ================
"""

def get_device():
    if args.device != "auto":
        return args.device
    try:
        import torch
        if torch.cuda.is_available():
            log.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return 0
    except Exception:
        pass
    log.info("GPU not available, using CPU")
    return "cpu"

DEVICE = get_device()

def get_classes():
    subdirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    if not subdirs:
        log.error("labal/ me koi subfolder nahi (salman_khan/, etc.)")
        return [], {}
    classes = []; class_map = {}
    for i, d in enumerate(subdirs):
        name = d.name.lower()
        classes.append(name); class_map[name] = i
    return classes, class_map

def show_stats(classes, class_map):
    print("\n--- Dataset Stats ---")
    print(f"{'Person':<20} {'Images':<8} {'Boxes':<8}")
    print("-" * 36)
    total_i = 0; total_b = 0
    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir(): continue
        name = subdir.name.lower()
        imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
        ni = len(imgs); nb = 0
        for img in imgs:
            lbl = img.with_suffix(".txt")
            if lbl.exists(): nb += sum(1 for _ in open(lbl) if _.strip())
        total_i += ni; total_b += nb
        print(f"{name:<20} {ni:<8} {nb:<8}")
    print("-" * 36)
    print(f"{'TOTAL':<20} {total_i:<8} {total_b:<8}")

def get_model_path():
    local = MODEL_DIR / args.model
    if local.exists():
        return str(local)
    if args.model == MODEL_FILE or MODEL_DIR.exists() and (MODEL_DIR / args.model).exists():
        pass
    return args.model

def auto_label(class_map):
    log.info("Auto-labeling with YOLO...")
    model = YOLO(get_model_path())
    all_imgs = []
    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir(): continue
        cid = class_map[subdir.name.lower()]
        imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
        if not imgs: continue
        from tqdm import tqdm
        for img_path in tqdm(imgs, desc=f"  {subdir.name}"):
            try:
                results = model(str(img_path), device=DEVICE, verbose=False)
            except Exception as e:
                log.warning(f"Failed {img_path.name}: {e}")
                continue
            lbl_path = img_path.with_suffix(".txt")
            with open(lbl_path, "w") as f:
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:
                        xywhn = box.xywhn[0].tolist()
                        f.write(f"{cid} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}\n")
            all_imgs.append(img_path)
    if not all_imgs:
        log.error("Koi image nahi mili.")
        return None
    log.info(f"Total: {len(all_imgs)} images labeled")
    return all_imgs

def augment_bbox(boxes, h, w, M, img_shape):
    import numpy as np
    new_boxes = []
    for box in boxes:
        xc, yc, bw, bh = box
        x1 = (xc - bw/2) * w
        y1 = (yc - bh/2) * h
        x2 = (xc + bw/2) * w
        y2 = (yc + bh/2) * h
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
        ones = np.ones((4, 1))
        xy = np.hstack([corners.reshape(4, 2), ones])
        transformed = M[:2] @ xy.T
        tx, ty = transformed[0], transformed[1]
        tx = np.clip(tx, 0, img_shape[1])
        ty = np.clip(ty, 0, img_shape[0])
        nx1, ny1 = tx.min(), ty.min()
        nx2, ny2 = tx.max(), ty.max()
        if nx2 - nx1 < 5 or ny2 - ny1 < 5:
            continue
        nx = (nx1 + nx2) / 2 / img_shape[1]
        ny = (ny1 + ny2) / 2 / img_shape[0]
        nw = (nx2 - nx1) / img_shape[1]
        nh = (ny2 - ny1) / img_shape[0]
        new_boxes.append([nx, ny, nw, nh])
    return new_boxes

def augment():
    log.info("Generating augmented variations...")
    from tqdm import tqdm
    import numpy as np
    methods = [
        ("flip", lambda img: cv2.flip(img, 1)),
        ("bright", lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=20)),
        ("dark", lambda img: cv2.convertScaleAbs(img, alpha=0.7, beta=-20)),
        ("rot15", lambda img: None),
        ("blur", lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
        ("noise", lambda img: None),
        ("scale_up", lambda img: None),
    ]
    count = 0
    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir(): continue
        imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
        for img_path in tqdm(imgs, desc=f"  {subdir.name}"):
            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists() or lbl_path.stat().st_size == 0: continue
            boxes = []; cids = []
            with open(lbl_path) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) == 5:
                        cids.append(int(p[0])); boxes.append(list(map(float, p[1:])))
            if not boxes: continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            h, w = img.shape[:2]
            for aug_name, fn in methods:
                aug_img = img.copy()
                h_aug, w_aug = h, w
                if aug_name == "rot15":
                    center = (w//2, h//2)
                    M = cv2.getRotationMatrix2D(center, random.uniform(-15, 15), 1.0)
                    aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    new_boxes = augment_bbox(boxes, h, w, M, aug_img.shape)
                    if len(new_boxes) != len(boxes) or not new_boxes:
                        continue
                    h_aug, w_aug = aug_img.shape[:2]
                elif aug_name == "noise":
                    noise = np.random.randint(0, 50, aug_img.shape, dtype=np.uint8)
                    aug_img = cv2.add(aug_img, noise)
                    new_boxes = [b[:] for b in boxes]
                elif aug_name == "scale_up":
                    scale = random.uniform(1.1, 1.3)
                    M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
                    aug_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    new_boxes = augment_bbox(boxes, h, w, M, aug_img.shape)
                    if len(new_boxes) != len(boxes) or not new_boxes:
                        continue
                    h_aug, w_aug = aug_img.shape[:2]
                else:
                    aug_img = fn(img)
                    new_boxes = [b[:] for b in boxes]
                aug_path = img_path.with_stem(f"{img_path.stem}_{aug_name}")
                cv2.imwrite(str(aug_path), aug_img)
                with open(aug_path.with_suffix(".txt"), "w") as f:
                    for c, b in zip(cids, new_boxes):
                        f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                count += 1
    log.info(f"Generated: {count} augmented images")

def preview(classes):
    log.info("Saving preview images...")
    pdir = BASE_DIR / "preview"; pdir.mkdir(exist_ok=True)
    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir(): continue
        cname = subdir.name.lower()
        try:
            cid = classes.index(cname)
        except ValueError:
            continue
        hue = cid * 0.618
        r, g, b = colorsys.hsv_to_rgb(hue % 1.0, 0.8, 0.9)
        color = (int(b*255), int(g*255), int(r*255))
        imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
        for img_path in imgs[:5]:
            lbl_path = img_path.with_suffix(".txt")
            if not lbl_path.exists() or lbl_path.stat().st_size == 0: continue
            img = cv2.imread(str(img_path))
            if img is None: continue
            hi, wi = img.shape[:2]
            with open(lbl_path) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) != 5: continue
                    _, xc, yc, bw, bh = map(float, p)
                    x1 = int((xc - bw/2) * wi); y1 = int((yc - bh/2) * hi)
                    x2 = int((xc + bw/2) * wi); y2 = int((yc + bh/2) * hi)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(cname, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(img, cname, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out = pdir / cname / img_path.name
            out.parent.mkdir(exist_ok=True); cv2.imwrite(str(out), img)
    log.info(f"Preview saved in {pdir}/")

def split_data(images):
    if TRAIN_DIR.exists(): shutil.rmtree(TRAIN_DIR)
    if VAL_DIR.exists(): shutil.rmtree(VAL_DIR)
    TRAIN_DIR.mkdir(parents=True); VAL_DIR.mkdir(parents=True)
    (TRAIN_DIR/"images").mkdir(); (TRAIN_DIR/"labels").mkdir()
    (VAL_DIR/"images").mkdir(); (VAL_DIR/"labels").mkdir()
    n = len(images)
    if n == 1:
        train_idx, val_idx = 1, 0
    else:
        random.shuffle(images)
        train_idx = max(1, n - max(1, n // 5))
        val_idx = n - train_idx
    for img in images[:train_idx]:
        shutil.copy(img, TRAIN_DIR/"images"/img.name)
        s = img.with_suffix(".txt")
        if s.exists(): shutil.copy(s, TRAIN_DIR/"labels"/f"{img.stem}.txt")
    if val_idx > 0:
        for img in images[train_idx:]:
            shutil.copy(img, VAL_DIR/"images"/img.name)
            s = img.with_suffix(".txt")
            if s.exists(): shutil.copy(s, VAL_DIR/"labels"/f"{img.stem}.txt")
    log.info(f"Split: Train={train_idx}, Val={val_idx}")

def make_yaml(classes):
    data = {"train": str(TRAIN_DIR/"images").replace("\\", "/"),
            "val": str(VAL_DIR/"images").replace("\\", "/"),
            "nc": len(classes), "names": classes}
    with open(DATA_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    log.info(f"Classes ({len(classes)}): {classes}")

def train(total_images):
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 100 if total_images < 10 else 50 if total_images < 100 else 30
    exp_name = args.name or f"yolo_run_{datetime.now().strftime('%m%d_%H%M')}"
    log.info(f"Training YOLO26 ({total_images} images, {epochs} epochs) on {DEVICE}...")
    model = YOLO(get_model_path())
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=args.imgsz,
        device=DEVICE,
        resume=args.resume,
        name=exp_name,
        patience=10,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        amp=False,
    )
    MODEL_DIR.mkdir(exist_ok=True)
    best_src = Path(f"runs/detect/{exp_name}/weights/best.pt")
    last_src = Path(f"runs/detect/{exp_name}/weights/last.pt")
    if best_src.exists():
        shutil.copy(best_src, MODEL_DIR / "best.pt")
        log.info(f"Best model saved: {MODEL_DIR / 'best.pt'}")
    if last_src.exists():
        shutil.copy(last_src, MODEL_DIR / "last.pt")
        log.info(f"Last checkpoint: {MODEL_DIR / 'last.pt'}")

def get_all_images():
    all_imgs = []
    for subdir in sorted(RAW_DIR.iterdir()):
        if not subdir.is_dir(): continue
        imgs = list(subdir.glob("*.jpg")) + list(subdir.glob("*.jpeg")) + list(subdir.glob("*.png"))
        imgs_with_labels = [p for p in imgs if p.with_suffix(".txt").exists()]
        all_imgs.extend(imgs_with_labels)
    return all_imgs

if __name__ == "__main__":
    print(logo)
    log.info(f"Device: {DEVICE}")
    classes, cmap = get_classes()
    if not classes:
        sys.exit(1)
    if not args.no_label:
        imgs = auto_label(cmap)
        if imgs is None:
            sys.exit(1)
    else:
        log.info("Skipping auto-labeling")
        imgs = get_all_images()
    show_stats(classes, cmap)
    n_orig = len(imgs)
    if not args.no_augment and n_orig < 50:
        augment()
    all_imgs = get_all_images()
    n_total = len(all_imgs)
    log.info(f"Total labeled images (after augment): {n_total}")
    preview(classes)
    split_data(all_imgs)
    make_yaml(classes)
    train(n_total)
