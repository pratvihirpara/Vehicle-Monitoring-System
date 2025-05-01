import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video file or webcam
video_path = "56310-479197605_small.mp4"  # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(5))

# Output video file
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Vehicle classes in COCO dataset
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Colors for different vehicle types (B,G,R)
colors = {
    "Car": (0, 255, 0),        # Green
    "Motorcycle": (0, 165, 255),   # Orange
    "Bus": (255, 0, 0),        # Blue
    "Truck": (0, 0, 255)       # Red
}

# Vehicle type counters
vehicle_type_counts = {vehicle_type: 0 for vehicle_type in vehicle_classes.values()}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)

    # Reset current frame type counts
    current_frame_type_counts = {vehicle_type: 0 for vehicle_type in vehicle_classes.values()}
    total_vehicles = 0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())

            if cls in vehicle_classes:
                vehicle_type = vehicle_classes[cls]
                color = colors[vehicle_type]
                
                # Increment counts
                total_vehicles += 1
                current_frame_type_counts[vehicle_type] += 1
                vehicle_type_counts[vehicle_type] += 1

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Overlay total vehicle count
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Display vehicle type counts
    y_offset = 70
    for vehicle_type, count in current_frame_type_counts.items():
        if count > 0:
            color = colors[vehicle_type]
            cv2.putText(frame, f"{vehicle_type}s: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
    
    # Show frame
    cv2.imshow("Traffic Monitoring", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
