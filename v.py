
import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in list(output_layers_indices)]

# Open a camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Create blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input to the network
    net.setInput(blob)

    # Run forward pass
    outs = net.forward(layer_names)

    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
      for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        
        try:
            confidence = scores[class_id]
        except IndexError:
            print("IndexError: Detection array has unexpected shape.")
            continue

        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
    
                

    # Apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes on the frame
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = classes[class_ids[i]]
        confidence = confidences[i]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
