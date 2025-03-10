from ultralytics import YOLO

# Load a pre-trained YOLOv11 model
model = YOLO('models/yolo11n.pt')  # or other YOLOv11 variants

# Display model information (optional)
model.info()

# Train the model on your custom dataset
# Shrimp-specific augmentations
results = model.train(
    data='data/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='small_shrimp_yolov11',
    device='cpu',
    workers=0,
    
    # Data augmentation parameters for shrimp detection
    augment=True,
    hsv_h=0.01,      # Slight hue variation (shrimp color is important)
    hsv_s=0.5,       # Moderate saturation changes
    hsv_v=0.3,       # Slight brightness changes
    degrees=5.0,     # Smaller rotation (shrimps often have specific orientations)
    translate=0.2,   # Allow more translation (shrimps can be anywhere)
    scale=0.4,       # Moderate scaling
    shear=0.0,       # No shear (preserves natural shape)
    perspective=0.0, # No perspective (underwater imagery often has consistent perspective)
    fliplr=0.5,      # Horizontal flip is fine
    flipud=0.0,      # Vertical flip rarely helps with aquatic creatures
    mosaic=0.8,      # High mosaic probability helps with small objects
    mixup=0.1,        # Light mixup can help with varying backgrounds

    plots=True,     # Plot results  (optional)
)

# Export the model
model.export(format='onnx')