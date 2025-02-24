import cv2 as cv
from ultralytics import YOLO
import imutils
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Define the y-coordinate for the reference line
REFERENCE_LINE_Y = 270

def processVideo():
    counter_cache = []
    detection_classes = []
    count = 0
    path = "videos/brine-shrimp.mp4" #360p video, so 640px (x) x 360px (y) resolution
    #read video
    vs = cv.VideoCapture(path)
    #load the model
    model = YOLO('models/bs-detect-model.pt')

    object_tracker = DeepSort(max_iou_distance=0.7,
                              max_age=5,
                              n_init=3,
                              nms_max_overlap=1.0,
                              max_cosine_distance=0.2,
                              nn_budget=None,
                              gating_only_position=False,
                              override_track_class=None,
                              embedder="mobilenet",
                              half=True,
                              bgr=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None
                              )

    # Dictionary to store the history of positions for each track ID
    track_history = {}

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # Modify the predict call to be more flexible
        results = model.predict(frame, stream=False, conf=0.25)  # Add confidence threshold
        
        # Debug print to see available classes
        print("Available classes:", results[0].names)
        detection_classes = results[0].names

        frame = draw_line(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box data
                x1, y1, x2, y2 = box.xyxy[0]  # Get box coordinates
                conf = box.conf[0]  # Get confidence
                cls = int(box.cls[0])  # Get class
                
                # Create data structure expected by drawBox
                data = [x1, y1, x2, y2, conf, cls]
                
                # Only draw if confidence is above threshold
                if conf > 0.25:
                    drawBox(data, frame, detection_classes[cls])
                    print(f"Detected class: {detection_classes[cls]} with confidence: {conf}")

            details = get_details(result, frame)
            tracks = object_tracker.update_tracks(details, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                break
            track_id = track.track_id
            bbox = track.to_ltrb()

            # Assigning Unique ID of the shrimp vehicle detected     
            cv.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1])), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0), 2)
            cv.putText(frame, "Dead shrimp count: " + str(count), (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                       2)  # Adjusted font size to 2 and thickness to 6

            # Update the track history
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append((int(bbox[0]), int(bbox[1])))
            
            #track line
            # Draw the trail
            for i in range(1, len(track_history[track_id])):
                cv.line(frame, track_history[track_id][i - 1], track_history[track_id][i], (255, 0, 0), 2)

            # Shrimp counter, comparing y axis of the detected shrimp to the reference line. cache the shrimp id to avoid multiple counting
            if bbox[1] > REFERENCE_LINE_Y and track_id not in counter_cache:
                counter_cache.append(track_id)
                count = count + 1
                cv.putText(frame, "Dead Shrimp: " + str(count), (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0),
                           2)  # Adjusted font size to 2 and thickness to 6

        # Show frames
        cv.imshow('image', frame)
        cv.waitKey(1)


def drawBox(data, image, name):
    x1, y1, x2, y2, conf, id = data
    p1 = (int(x1), int(y1))
    p2 = (int(x2), int(y2))
    cv.rectangle(image, p1, p2, (0, 0, 255), 3)
    cv.putText(image, name, p1, cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

    return image

def get_details(result, image):
    # Modified to handle different YOLO model outputs
    boxes = result.boxes
    detections = []
    
    for box in boxes:
        xywh = box.xywh[0].cpu().numpy()  # Get box in xywh format
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        
        sample = (xywh, conf, cls)
        detections.append(sample)

    return detections

# Draw line on the frame as reference point of tracker, line depth is adjusted as reference point (REFERENCE_LINE_Y)
def draw_line(image):
    p1 = (600, REFERENCE_LINE_Y) #p1 for x coord of line, 
    p2 = (image.shape[1] - 600, REFERENCE_LINE_Y) #p2 for -x coord of line (consider as number line/garis bilangan)
    print(p1, p2)
    image = cv.line(image, p1, p2, (0, 255, 0), thickness=2)  # Adjusted thickness to 2

    return image

processVideo()






