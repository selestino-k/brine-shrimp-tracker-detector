from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolo11n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
model.train(data="one-class-bslt-data.yaml", epochs=100, imgsz=640, device='cpu', workers=0)

#results= model("brine-shrimp.mp4", save=True, show=True)

