import cv2
import numpy as np
import urllib.request
import face_recognition
# Load a sample image with known faces
known_image = face_recognition.load_image_file("C:/Users/Pradeep kumar mahato/Pictures/Camera Roll/WIN_20240117_20_45_26_Pro.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
# Download YOLO files
yolo_files = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg": "https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true",
    "coco.names": "https://github.com/pjreddie/darknet/blob/master/data/coco.names?raw=true"
}

for file, url in yolo_files.items():
    urllib.request.urlretrieve(url, file)

# Rest of the code remains the same...
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in list(output_layers_indices)]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f]

# Open the camera
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Use '0' for the default camera. Change to '1' if you have an additional camera.

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    height, width, channels = frame.shape
    #face dedect
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches the known face
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        name = "Unknown"
        if matches[0]:
            name = "Jagdeep"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to draw bounding box
    class_ids = []
    confidences = []
    boxes = []

    # Loop through each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 :#and class_id == classes.index("person"):
                # Object detected is a face
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow("YOLO Face Detection", frame)
#     cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
