import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort
import customtkinter as ctk
from settings import *
from PIL import Image, ImageTk
import tkinter as tk
import threading
from sqlite import BottleDB
import os

# Load YOLO models (một lần duy nhất)
det_model = YOLO(r"E:\OneDrive - vnu.edu.vn\Desktop\GitHub\Do_An\runs\detect\train4\weights\best.pt")
cls_model = YOLO(r"E:\OneDrive - vnu.edu.vn\Desktop\GitHub\Do_An\runs\classify\train3\weights\best.pt")

# Set device và optimize
det_model.to('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')
cls_model.to('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')


class VidFrame(ctk.CTkFrame):
    def __init__(self, parent, video_source, stats_callback=None, db=None, uart_callback=None):
        super().__init__(parent, fg_color=BACKGROUND_COLOR)
        
        self.video_source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.stats_callback = stats_callback
        self.db = db
        self.uart_callback = uart_callback

        # folder luu chai
        self.save_dir = os.path.join(os.path.dirname(__file__), "captures")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Giảm buffer size về 1 <mặc định opencv buffer 5-10 frame trong RAM>
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Lấy kích thước video
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Vùng trung tâm (cache)
        self.center_x_min = int(self.frame_width * 0.4)
        self.center_x_max = int(self.frame_width * 0.6)
        
        # Init tracker
        self.tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Status và stats
        self.bottle_status = {}
        self.stats = {
            'total': 0,
            'ok': 0,
            'water_error': 0,
            'label_error': 0,
            'current_bottles': []
        }
        
        # Canvas
        self.video_canvas = tk.Canvas(
            self,
            bg=BACKGROUND_COLOR,
            highlightthickness=0
        )
        self.video_canvas.pack(fill='both', expand=True)
        
        # Threading
        self.running = True
        self.current_image = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        
        # Cache label classification
        self.label_cache = {}  # {(x1,y1,x2,y2): (result, frame_count)}
        self.cache_lifetime = 5  # Giữ cache 5 frames
        self.current_frame_count = 0
        
        # Frame processing settings
        self.frame_skip = 2  # Xử lý mỗi 2 frames (tăng FPS gấp đôi)
        self.frame_counter = 0
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start UI update
        self.update_display()
    
    def _capture_loop(self):
        """Background thread để capture và xử lý frame"""
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame = cv2.flip(frame, 1)
            
            # SKIP FRAMES để tăng FPS
            self.frame_counter += 1
            if self.frame_counter % self.frame_skip != 0:
                # Skip frame này, dùng lại kết quả cũ
                continue
            
            # Xử lý frame
            processed = self._process_frame(frame)
            
            # Cập nhật processed frame (thread-safe)
            with self.frame_lock:
                self.processed_frame = processed
    
    def _process_frame(self, frame):
        """Xử lý frame với YOLO - tối ưu"""
        self.current_frame_count += 1
        
        # Detection (batch = 1, verbose = False để tăng tốc)
        results = det_model(frame, verbose=False)[0]
        detections = []
        label_info = []

        # Extract detections
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        for box, cls_id, conf in zip(boxes, classes, confs):
            cls_id = int(cls_id)
            class_name = det_model.names[cls_id]
            x1, y1, x2, y2 = map(int, box)

            if class_name == 'label':
                label_crop = frame[y1:y2, x1:x2]
                if label_crop.size == 0:
                    continue

                # TỐI ƯU: LABEL CACHE
                bbox_key = (x1//10, y1//10, x2//10, y2//10)
                
                # Check cache
                if bbox_key in self.label_cache:
                    cached_result, cached_frame = self.label_cache[bbox_key]
                    if self.current_frame_count - cached_frame < self.cache_lifetime:
                        # Dùng kết quả cache
                        label_ok = cached_result
                        color = (0, 255, 0) if label_ok else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_info.append(((x1, y1, x2, y2), label_ok))
                        continue
                
                # Không có cache → classify mới
                try:
                    # Resize và classify
                    label_resized = cv2.resize(label_crop, (224, 224))
                    label_resized = cv2.cvtColor(label_resized, cv2.COLOR_BGR2RGB)

                    cls_result = cls_model(label_resized, verbose=False)[0]
                    probs = cls_result.probs.data.cpu().numpy()
                    cls_idx = np.argmax(probs)
                    cls_label = cls_model.names[cls_idx]

                    label_ok = (cls_label == "label_ok")
                    
                    # Lưu vào cache
                    self.label_cache[bbox_key] = (label_ok, self.current_frame_count)
                    
                    color = (0, 255, 0) if label_ok else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    label_info.append(((x1, y1, x2, y2), label_ok))
                except:
                    pass

            elif class_name == 'bottle':
                detections.append([x1, y1, x2, y2, conf])
        
        # Cleanup cache cũ mỗi 50 frames
        if self.current_frame_count % 50 == 0:
            self._cleanup_cache()
        
        # Tracking
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        tracks = self.tracker.update(detections)
        
        # Process tracked bottles
        current_ids = []
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            
            current_ids.append(track_id)
            
            # Init status
            if track_id not in self.bottle_status:
                self.bottle_status[track_id] = {
                    'water_ok': True,
                    'label_ok': True,
                    'error_type': None,
                    'counted': False,
                    'finalized': False
                }
    
            # VẼ BBOX BOTTLE TRƯỚC (luôn vẽ)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # CHECK WATER LEVEL (luôn vẽ)
            water_ok = self._check_water_level(frame, (x1, y1, x2, y2))
            
            # Chỉ update status nếu chưa finalize
            if not self.bottle_status[track_id]['finalized']:
                # Check label overlap
                label_ok = True
                for (lx1, ly1, lx2, ly2), l_ok in label_info:
                    if self._check_overlap((x1, y1, x2, y2), (lx1, ly1, lx2, ly2)):
                        label_ok = l_ok
                        break
                
                self.bottle_status[track_id]['water_ok'] = water_ok
                self.bottle_status[track_id]['label_ok'] = label_ok
                
                # Chốt ở center zone
                if self._is_in_center((x1, y1, x2, y2)):
                    self._finalize_bottle(track_id, frame,(x1, y1, x2, y2))
        
        # Update stats
        self._update_stats(current_ids)
        
        # Cleanup
        for bottle_id in list(set(self.bottle_status.keys()) - set(current_ids)):
            del self.bottle_status[bottle_id]
        
        return frame
    
    def _cleanup_cache(self):
        """Xóa cache cũ"""
        current = self.current_frame_count
        self.label_cache = {
            k: v for k, v in self.label_cache.items()
            if current - v[1] < self.cache_lifetime
        }
    
    def _check_water_level(self, frame, bottle_box):
        """Kiểm tra mực nước - tối ưu"""
        x1, y1, x2, y2 = map(int, bottle_box)
        h = y2 - y1
        # Clamp bounding box to frame
        x1c = max(0, x1)
        x2c = min(self.frame_width, x2)
        y1c = max(0, y1)
        y2c = min(self.frame_height, y2)
        h_c = y2c - y1c
        if h_c <= 0 or x2c <= x1c:
            return False, (255, 255, 255)

        roi_y = y1c + int(h_c * 0.15)
        roi_h = int(h_c * 0.25)
        roi_y2 = min(self.frame_height, roi_y + roi_h)
        roi = frame[roi_y:roi_y2, x1c:x2c]

        if roi.size == 0:
            return False, (255, 255, 255)
    
        # Tối ưu: dùng gray trực tiếp, không blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # ksize=3 nhanh hơn
        mean_gradient = np.mean(np.abs(sobel_y), axis=1)

        line_offset = np.argmax(mean_gradient)
        detected_y = roi_y + line_offset
        expected_y = roi_y + int((roi_y2 - roi_y) / 2)

        water_ok = detected_y <= expected_y + 8
        color = (0, 255, 0) if water_ok else (0, 0, 255)

        # Clamp detected_y to frame
        detected_y = min(max(detected_y, 0), self.frame_height - 1)
        cv2.line(frame, (x1c, detected_y), (x2c, detected_y), color, 2)
        return water_ok
    
    def _check_overlap(self, box1, box2):
        """Kiểm tra overlap - tối ưu"""
        bx1, by1, bx2, by2 = box1
        lx1, ly1, lx2, ly2 = box2
        
        ix1, iy1 = max(bx1, lx1), max(by1, ly1)
        ix2, iy2 = min(bx2, lx2), min(by2, ly2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        label_area = (lx2 - lx1) * (ly2 - ly1)
        
        return (inter_area / label_area) > 0.5
    
    def _is_in_center(self, box):
        """Check center zone - cached"""
        center_x = (box[0] + box[2]) / 2
        return self.center_x_min <= center_x <= self.center_x_max
    
    def _finalize_bottle(self, track_id, frame, box):
        """Chốt kết quả bottle"""
        self.bottle_status[track_id]['finalized'] = True
        
        w_ok = self.bottle_status[track_id]['water_ok']
        l_ok = self.bottle_status[track_id]['label_ok']
        
        if w_ok and l_ok:
            error_type = 0
            self.stats['ok'] += 1
        elif not w_ok and l_ok:
            error_type = 1
            self.stats['water_error'] += 1
        else:
            error_type = 2
            self.stats['label_error'] += 1
        
        self.bottle_status[track_id]['error_type'] = error_type
        self.bottle_status[track_id]['counted'] = True
        self.stats['total'] += 1

        # Save captured image
        # Save cropped bottle image as "<ID>_<mã lỗi>.jpg"
        x1, y1, x2, y2 = map(int, box)
        x1c = max(0, min(x1, self.frame_width - 1))
        y1c = max(0, min(y1, self.frame_height - 1))
        x2c = max(0, min(x2, self.frame_width))
        y2c = max(0, min(y2, self.frame_height))
        if x2c > x1c and y2c > y1c:
            crop = frame[y1c:y2c, x1c:x2c]
            if crop.size > 0:
                filename = f"ID-{track_id}_TYPE-{error_type}.jpg"
                save_path = os.path.join(self.save_dir, filename)
                try:
                    cv2.imwrite(save_path, crop)
                except Exception:
                    pass

        # Sqlite insert 
        if self.db:
            try:
                status = 'OK' if error_type == 0 else 'ERROR'
                self.db.insert_data(
                    bottle_id=str(track_id),
                    status=status,
                    error_type=str(error_type)
                )
            except:
                pass

        # UART callback
        if self.uart_callback:
            try:
                self.uart_callback(str(error_type))
            except:
                pass
        
    
    def _update_stats(self, current_ids):
        """Update current bottles list"""
        self.stats['current_bottles'] = [
            (tid, info['water_ok'], info['label_ok'], info.get('error_type', -1))
            for tid, info in self.bottle_status.items()
            if tid in current_ids
        ]
        
        if self.stats_callback:
            self.stats_callback(self.stats)
    
    def update_display(self):
        """Update UI - chạy ở main thread"""
        if not self.running:
            return
        
        # Lấy processed frame (thread-safe)
        with self.frame_lock:
            if self.processed_frame is not None:
                frame_to_display = self.processed_frame.copy()
            else:
                self.after(16, self.update_display)  # ~60 FPS
                return
        
        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_ratio = pil_img.width / pil_img.height
            canvas_ratio = canvas_width / canvas_height
            
            if canvas_ratio > img_ratio:
                new_height = canvas_height
                new_width = int(new_height * img_ratio)
            else:
                new_width = canvas_width
                new_height = int(new_width / img_ratio)
            
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.BILINEAR)  # BILINEAR nhanh hơn LANCZOS
        
        # Display
        self.current_image = ImageTk.PhotoImage(pil_img)
        self.video_canvas.delete("all")
        
        x = (canvas_width - pil_img.width) // 2
        y = (canvas_height - pil_img.height) // 2
        
        self.video_canvas.create_image(x, y, anchor='nw', image=self.current_image)
        
        # Loop 
        self.after(100, self.update_display)
    
    def stop(self):
        """Dừng threads"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        self.stop()