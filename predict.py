import sys, yaml, argparse, logging, time
from pathlib import Path
from ultralytics import YOLO
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

parser = argparse.ArgumentParser(description="YOLO Inference")
parser.add_argument("source", nargs="?", default=None, help="Image path, directory, video, or glob pattern")
parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold (default: 0.45)")
parser.add_argument("--device", type=str, default="auto", help="Device: cpu, 0, or auto")
parser.add_argument("--save", action="store_true", default=True, help="Save output images")
parser.add_argument("--show", action="store_true", help="Display results in a window")
parser.add_argument("--model", type=str, default=None, help="Path to custom model .pt file")
args = parser.parse_args()

def get_device():
    if args.device != "auto":
        return args.device
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"

DEVICE = get_device()

with open(BASE_DIR / "data.yaml") as f:
    names = yaml.safe_load(f)["names"]

model_path = args.model
if not model_path:
    best = BASE_DIR / "models/best.pt"
    if best.exists():
        model_path = str(best)
    else:
        best = BASE_DIR / "runs/detect/train/weights/best.pt"
        if best.exists():
            model_path = str(best)
        else:
            log.error("best.pt nahi mila. Pehle python train.py run karo.")
            sys.exit(1)

log.info(f"Loading model: {model_path}")
model = YOLO(model_path)

source = args.source
if not source:
    source = input("Image/video path: ").strip()

source = Path(source)
if not source.exists():
    log.error(f"{source} does not exist")
    sys.exit(1)

color_map = {}
for i in range(len(names)):
    import colorsys
    h = i * 0.618
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.7, 0.85)
    color_map[i] = (int(b * 255), int(g * 255), int(r * 255))

def draw_boxes(img, results, names, color_map):
    for box in results.boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < args.conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = color_map.get(cid, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cid]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img

if source.is_dir():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(source.glob(ext))
    if not files:
        log.error(f"No images found in {source}")
        sys.exit(1)
    out_dir = BASE_DIR / "output" / source.name
    out_dir.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm
    log.info(f"Processing {len(files)} images from {source}...")
    for f in tqdm(files, desc="Predict"):
        results = model(str(f), device=DEVICE, conf=args.conf, iou=args.iou, verbose=False)
        img = cv2.imread(str(f))
        if img is None: continue
        img = draw_boxes(img, results[0], names, color_map)
        out_path = out_dir / f.name
        cv2.imwrite(str(out_path), img)
    log.info(f"Saved to {out_dir}/")
    sys.exit(0)

elif source.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
    log.info(f"Predicting: {source}")
    results = model(str(source), device=DEVICE, conf=args.conf, iou=args.iou, verbose=False)
    img = cv2.imread(str(source))
    if img is not None:
        img = draw_boxes(img, results[0], names, color_map)
    print(f"\n{'Person':<22} {'Confidence':<10}")
    print("-" * 32)
    for box in results[0].boxes:
        cid = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"{names[cid]:<22} {conf:.1%}")
    if args.save:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_name = f"output_{ts}.jpg"
        if img is not None:
            cv2.imwrite(out_name, img)
        else:
            results[0].save(out_name)
        log.info(f"Saved: {out_name}")
    if args.show and img is not None:
        cv2.imshow("YOLO Predict", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

elif source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
    log.info(f"Processing video: {source}")
    cap = cv2.VideoCapture(str(source))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"output_{ts}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_name, fourcc, fps, (w, h))
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="Video")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, device=DEVICE, conf=args.conf, iou=args.iou, verbose=False)
        frame = draw_boxes(frame, results[0], names, color_map)
        out.write(frame)
        if args.show:
            cv2.imshow("YOLO Predict", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        pbar.update(1)
    cap.release(); out.release(); pbar.close()
    log.info(f"Video saved: {out_name}")
    cv2.destroyAllWindows()

else:
    log.error(f"Unsupported file: {source}")
    sys.exit(1)
