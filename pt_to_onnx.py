from ultralytics import YOLO

model_path = r"E:\OneDrive - vnu.edu.vn\Desktop\GitHub\Do_An\runs\classify\train3\weights\best.pt"

model = YOLO(model_path)

model.export(
    format="onnx",
    opset=12,                # Phiên bản ONNX, khuyến nghị 12 hoặc 13
    simplify=True,           # Tối ưu ONNX model (xóa các node dư thừa)
    dynamic=True,            # Hỗ trợ kích thước ảnh linh hoạt
    imgsz=640                # Kích thước input ảnh
)