import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class EngineInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine_data = f.read()
        
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_data)
        self.context = self.engine.create_execution_context()
        
        # Allocate memory for input and output
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': cuda_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': cuda_mem})

    def infer(self, img):
        # Process input
        self.inputs[0]['host'] = np.ravel(img)
        
        # Transfer input data to device
        for inp in self.inputs:
            cuda.memcpy_htod(inp['device'], inp['host'])
        
        # Run inference
        self.context.execute_v2(self.bindings)
        
        # Transfer predictions back
        for out in self.outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])
        
        return [out['host'] for out in self.outputs]

def check_water_level_dynamic(frame, bottle_box, expected_offset_ratio=0.15, roi_height_ratio=0.25, threshold=8):
    x1, y1, x2, y2 = map(int, bottle_box)
    w = x2 - x1
    h = y2 - y1

    roi_x = x1
    roi_w = w
    roi_y = y1 + int(h * expected_offset_ratio)
    roi_h = int(h * roi_height_ratio)

    roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    if roi.size == 0:
        return

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    mean_gradient = np.mean(np.abs(sobel_y), axis=1)
    
    line_offset = np.argmax(mean_gradient)
    detected_y = roi_y + line_offset
    expected_y = roi_y + int(roi_h / 2)

    status = "OK" if detected_y <= expected_y + threshold else "LOW"
    color = (0, 255, 0) if status == "OK" else (0, 0, 255)
    
    cv2.line(frame, (roi_x, detected_y), (roi_x + roi_w, detected_y), color, 2)

def process_detections(output, input_shape, orig_shape, conf_thres=0.25, iou_thres=0.45):
    """
    Process YOLO model output
    Args:
        output: Raw output from YOLO model
        input_shape: Model input shape (height, width)
        orig_shape: Original image shape (height, width)
        conf_thres: Confidence threshold
        iou_thres: NMS IoU threshold
    """
    # Reshape output to [num_boxes, num_classes + 5]
    # YOLO outputs: [x, y, w, h, conf, class_scores]
    predictions = output[0].reshape(-1, 7)  # Adjust 7 based on your num_classes + 5
    
    # Filter by confidence
    conf = predictions[:, 4]
    mask = conf > conf_thres
    predictions = predictions[mask]
    
    if not len(predictions):
        return [], [], []
    
    # Get boxes, scores and classes
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    class_ids = predictions[:, 5].astype(np.int32)
    
    # Convert boxes to corners format (xywh to xyxy)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    # Scale boxes to original image size
    scale_x = orig_shape[1] / input_shape[1]
    scale_y = orig_shape[0] / input_shape[0]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        return boxes, scores, class_ids
    
    return [], [], []

def main():
    # Đường dẫn tới các file engine và video input
    DET_ENGINE_PATH = "path/to/your/detection.engine"  # Thay đổi đường dẫn tới detection engine
    CLS_ENGINE_PATH = "path/to/your/classification.engine"  # Thay đổi đường dẫn tới classification engine
    INPUT_SOURCE = 0  # 0 cho webcam, hoặc đường dẫn đến file video/ảnh
    
    # Load TensorRT engines
    print(f"Loading detection engine from: {DET_ENGINE_PATH}")
    det_engine = EngineInference(DET_ENGINE_PATH)
    
    print(f"Loading classification engine from: {CLS_ENGINE_PATH}")
    cls_engine = EngineInference(CLS_ENGINE_PATH)
    
    # Setup video capture
    if isinstance(INPUT_SOURCE, str):
        print(f"Opening video file: {INPUT_SOURCE}")
        cap = cv2.VideoCapture(INPUT_SOURCE)
    else:
        print("Opening webcam")
        cap = cv2.VideoCapture(INPUT_SOURCE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Prepare input for detection
        input_size = (640, 640)  # Adjust based on your model
        img = cv2.resize(frame, input_size)
        img = img.astype(np.float32) / 255.0  # Normalize
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        
        # Run detection
        det_output = det_engine.infer(img)
        boxes, scores, class_ids = process_detections(det_output, frame.shape)
        
        # Process each detection
        for box, score, class_id in zip(boxes, scores, class_ids):
            if score < 0.5:  # Confidence threshold
                continue
                
            x1, y1, x2, y2 = map(int, box)
            
            if class_id == 1:  # Label class
                # Crop and process label
                label_crop = frame[y1:y2, x1:x2]
                if label_crop.size == 0:
                    continue
                    
                try:
                    # Prepare input for classification
                    label_resized = cv2.resize(label_crop, (224, 224))
                    label_input = np.transpose(label_resized, (2, 0, 1)).astype(np.float32) / 255.0
                    
                    # Run classification
                    cls_output = cls_engine.infer(label_input)
                    cls_idx = np.argmax(cls_output[0])
                    
                    # Draw results
                    color = (0, 255, 0) if cls_idx == 0 else (0, 0, 255)  # Assuming 0 is label_ok
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                except Exception as e:
                    print(f"Classification error: {e}")
                    continue
                    
            elif class_id == 0:  # Bottle class
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                check_water_level_dynamic(frame, (x1, y1, x2, y2))
        
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
