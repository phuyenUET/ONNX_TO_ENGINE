import cv2
import numpy as np
import onnxruntime as ort
import time

# ====== 1. Load model ======
onnx_path = r"E:\OneDrive - vnu.edu.vn\Desktop\GitHub\Do_An\runs\detect\train4\weights\best.onnx"
session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ====== 2. Đọc ảnh test ======
img_path = r"E:\OneDrive - vnu.edu.vn\Desktop\GitHub\Do_An\lan2_train_data\data\images\train\frame_00026_jpg.rf.b9db8874621000e0e0b3534891a5cfc6.jpg"  # ảnh có chai nước
img = cv2.imread(img_path)
h0, w0 = img.shape[:2]

# Resize về (640,640)
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC->CHW
img_input = np.expand_dims(img_input, axis=0).astype(np.float32) / 255.0

# ====== 3. Inference ======
start = time.time()
raw_outputs = session.run([output_name], {input_name: img_input})[0]
end = time.time()
print(f"Inference time: {(end-start)*1000:.2f} ms")

# ====== 4. Giải mã kết quả (YOLO v8/v11: [x,y,w,h, cls0..clsN]) ======
arr = raw_outputs[0]
# Đưa về dạng (N, no)
pred = arr.T if arr.shape[0] <= arr.shape[1] else arr
no = pred.shape[1]
nc = no - 4
if nc < 1:
    raise RuntimeError(f"Đầu ra ONNX không hợp lệ cho YOLOv8/v11 (no={no}). Kỳ vọng no >= 5.")


conf_thres = 0.4
iou_thres = 0.5

# Màu và tên lớp
colors = {
    0: (0, 255, 0),    # bottle - xanh lá
    1: (255, 0, 0)     # label - đỏ
}
names = {
    0: "bottle",
    1: "label"
}

def iou_xyxy(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + areas - inter + 1e-6
    return inter / union

def nms(boxes, scores, iou_thr=0.5):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_thr]
    return keep

dets = []
for p in pred:
    x, y, w, h = float(p[0]), float(p[1]), float(p[2]), float(p[3])

    # YOLO v8/v11: [x,y,w,h, cls...]
    cls_scores = p[4:]
    cls_id = int(np.argmax(cls_scores))
    cls_conf = float(cls_scores[cls_id])

    if cls_conf < conf_thres:
        continue

    # center->corner, scale về ảnh gốc
    x1 = int((x - w / 2) * w0 / 640)
    y1 = int((y - h / 2) * h0 / 640)
    x2 = int((x + w / 2) * w0 / 640)
    y2 = int((y + h / 2) * h0 / 640)

    # Clip về biên ảnh
    x1 = max(0, min(x1, w0 - 1))
    y1 = max(0, min(y1, h0 - 1))
    x2 = max(0, min(x2, w0 - 1))
    y2 = max(0, min(y2, h0 - 1))

    dets.append([x1, y1, x2, y2, cls_conf, cls_id])

dets = np.array(dets, dtype=np.float32)
if dets.size > 0:
    boxes_xyxy = dets[:, :4]
    scores = dets[:, 4]
    cls_ids = dets[:, 5].astype(int)
    keep = nms(boxes_xyxy, scores, iou_thres)

    # ====== 5. Vẽ bbox ======
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        conf = scores[i]
        cls_id = int(cls_ids[i])
        color = colors.get(cls_id, (0, 255, 255))
        label = names.get(cls_id, f"class {cls_id}")
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
else:
    print("Không có detection nào vượt ngưỡng.")

# ====== 6. Hiển thị ảnh ======
cv2.imshow("YOLO ONNX Output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
