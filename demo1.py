import cv2
import face_recognition

# Load a sample image with known faces
known_image = face_recognition.load_image_file("C:/Users/Pradeep kumar mahato/Pictures/Camera Roll/WIN_20240117_20_45_26_Pro.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Open a camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the camera
    ret, frame = cap.read()

    # Find all face locations and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches the known face
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        name = "Unknown"
        if matches[0]:
            name = "Known Person"

        # Draw rectangle around the face and display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the result
    cv2.imshow("Face Recognition", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
