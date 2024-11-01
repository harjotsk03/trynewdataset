from ultralytics import YOLO

# Load a YOLOv8 model (e.g., YOLOv8n for nano version, YOLOv8s for small)
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', 'yolov8m.pt', etc.

# Train the model
model.train(
    data=r"C:\Users\harjo\OneDrive\Desktop\trainmodeliat\data.yaml",
    epochs=50,          # Number of epochs to train
    imgsz=640,          # Image size (can also use 416 or 1280 depending on requirements)
    batch=64,           # Batch size
    name='yolov8-waste', # Name for saving training results
    val=True            # Perform validation during training
)

# Optional: Evaluate the model on the test set after training
metrics = model.val(data = r"C:\Users\harjo\OneDrive\Desktop\trainmodeliat\data.yaml")

print("Evaluation Metrics:", metrics)
