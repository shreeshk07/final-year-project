import cv2
import os

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a folder to store the images
if not os.path.exists('images'):
    os.makedirs('images')

# Capture and save images
i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Save the image
    filename = 'images/{}.jpg'.format(i)
    cv2.imwrite(filename, frame)

    i += 1

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
