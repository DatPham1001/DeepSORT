import cv2
# from yolov5  import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from yolov5.models.common import AutoShape, DetectMultiBackend
# from utils.torch_utils import select_device

# Load YOLOv5 locally
# model = YOLO("weights/yolov5s.pt")  # Replace with your local model path
video_path = "data/test.mp4"
tracking_class = 2 # None: track all

filter_classes = [2, 3, 7]  # Ví dụ: 2 = car, 3 = motorcycle, 7 = truck
conf_threshold = 0.5
cap = cv2.VideoCapture(video_path)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))


#Config deepSORT
tracker = DeepSort(max_age=30, nn_budget=100)

#Tạo model yolo

device = "cpu"
model  = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True )
model  = AutoShape(model)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Detect objects
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    # Lọc các object theo filter_classes
    filtered_detections = [det for det in detections if int(det[5]) in filter_classes]

    # Chuẩn bị dữ liệu cho DeepSORT
    bbox_xywh = []
    confidences = []
    class_ids = []
    for det in filtered_detections:
        x1, y1, x2, y2, conf, cls = det
        bbox_xywh.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        confidences.append(conf)
        class_ids.append(int(cls))
    # Cập nhật tracker
    tracks = tracker.update_tracks(bbox_xywh, confidences, class_ids, frame=frame)
    # Vẽ bounding box và ID
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)

        # Vẽ bounding box và ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Ghi vào file output
    out.write(frame)
    # Display the frame
    cv2.imshow('YOLOv5 + DeepSORT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
