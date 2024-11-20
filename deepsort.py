import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Danh sách các lớp cần theo dõi (dựa trên COCO classes)
filter_classes = [2, 3, 7]  # Car, Motorcycle, Truck

# Kiểm tra và thiết lập thiết bị
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# Đọc danh sách class từ file
with open("classname.txt", "r") as f:
    class_names = f.read().strip().split("\n")

# Load YOLOv8 model
model = YOLO('weights/yolo11s.pt')  # Sử dụng mô hình YOLOv8 (tiny model)
model.to(device)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Khởi tạo DeepSORT
tracker = DeepSort(max_age=30, nn_budget=100)

# Mở video
video_path = "data/test.mp4"
cap = cv2.VideoCapture(video_path)

# Tạo output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Dự đoán với YOLOv8
    results = model.predict(frame, conf=0.25, iou=0.45)
    detections = results[0].boxes.data.cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    # Lọc object theo filter_classes
    filtered_detections = [det for det in detections if int(det[5]) in filter_classes]

    # Chuyển đổi định dạng cho DeepSORT
    bbox_xywh = []
    confidences = []
    class_ids = []
    detect = []
    for det in filtered_detections:
        x1, y1, x2, y2, conf, cls = det
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confidences.append(conf)
        class_id= int(cls)
        class_ids.append(class_id)
        detect.append([ [x1, y1, x2-x1, y2 - y1], conf, class_id ])

    # Cập nhật tracker
    tracks = tracker.update_tracks(detect, frame=frame)

    # Vẽ bounding box và ID
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)
        # Lấy tên lớp từ class_names
        class_id = track.det_class
        class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

        # Vẽ bounding box và ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Ghi vào file output
    out.write(frame)

    # Hiển thị khung hình
    cv2.imshow('YOLOv8 + DeepSORT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()
