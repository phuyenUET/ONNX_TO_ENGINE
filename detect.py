import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from pathlib import Path

# ==== CONFIG ====
engine_path = "best_alpha_IoU.engine"
img_dir = Path("test_images")
output_dir = Path("detect_results")
output_dir.mkdir(exist_ok=True)

conf_threshold = 0.3
nms_threshold = 0.45
input_w, input_h = 640, 640
class_names = ["port", "ship"]

# ==== LOAD ENGINE ====
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# ==== BINDING ====
input_shape = engine.get_binding_shape(0)
output_shape = engine.get_binding_shape(1)
print(f"Input shape: {input_shape}, Output shape: {output_shape}")

# ==== GPU BUFFER ====
input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize
d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
bindings = [int(d_input), int(d_output)]

# ==== HÀM NMS ====
def nms(boxes, confs, iou_thresh=0.45):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    confs = np.array(confs)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = confs.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# ==== LOOP QUA ẢNH ====
for img_path in sorted(img_dir.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    orig_h, orig_w = img.shape[:2]

    # === PREPROCESS ===
    img_resized = cv2.resize(img, (input_w, input_h))
    img_input = img_resized[..., ::-1].transpose(2, 0, 1)
    img_input = np.ascontiguousarray(img_input, dtype=np.float32) / 255.0
    img_input = np.expand_dims(img_input, 0)

    # === COPY TO GPU ===
    cuda.memcpy_htod(d_input, img_input)

    # === INFERENCE ===
    context.execute_v2(bindings)
    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output, d_output)
    output = np.squeeze(output)

    # === DECODE ===
    boxes = output[:4, :].T
    scores = output[4:, :].T
    class_ids = np.argmax(scores, axis=1)
    conf = np.max(scores, axis=1)

    # xywh → xyxy
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # scale về ảnh gốc
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    # === FILTER & NMS ===
    mask = conf > conf_threshold
    boxes_xyxy = boxes_xyxy[mask]
    conf = conf[mask]
    class_ids = class_ids[mask]
    keep = nms(boxes_xyxy, conf, nms_threshold)

    # === DRAW ===
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cls_id = int(class_ids[i])
        label = f"{class_names[cls_id]}: {conf[i]:.2f}"
        color = (0, 255, 0) if cls_id == 1 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = output_dir / img_path.name
    cv2.imwrite(str(out_path), img)
    print(f"✅ Done: {img_path.name}")

print("🎯 All images processed!")

