import cv2
import numpy as np
from tkinter import filedialog
filename = filedialog.askopenfilename(title='open')
# Load the image
img = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to the image
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# Find contours in the image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter the contours based on their shape and size
eye_contours = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if area > 50 and perimeter > 150:
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        if len(approx) == 4:
            eye_contours.append(cnt)

# Apply Hough transform to detect circles in the eye-like shapes
circles = []
for cnt in eye_contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    if radius > 10 and radius < 50:
        circles.append((center, radius))

# Draw the detected circles on the original image
for center, radius in circles:
    cv2.circle(img, center, radius, (0, 255, 0), 2)

# Display the image with detected circles
cv2.imshow('Image with Circles Detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define color ranges for different eye colors in HSV color space
color_ranges = {
    'blue': [(100, 50, 50), (130, 255, 255)],
    'black': [(30, 50, 50), (80, 255, 255)],
    'brown': [(0, 50, 50), (30, 255, 255)]
}

# Threshold the image based on the defined color ranges
eye_color = None
for color, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    count = cv2.countNonZero(mask)
    if count > 0:
        eye_color = color
        break

# Display the detected eye color
if eye_color:
    print("The detected eye color is: ", eye_color)
else:
    print("No eye color detected.")


##hair colour

import cv2
import numpy as np
from tkinter import filedialog
#filename = filedialog.askopenfilename(title='open')
# Load the image
img = cv2.imread(filename)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a median blur to remove noise
gray = cv2.medianBlur(gray, 5)

# Detect edges using the Canny edge detector
edges = cv2.Canny(gray, 50, 150)

# Perform morphological operations to close gaps in the edges
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours in the closed image
contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area (assumed to be the hair)
max_area = 0
max_contour = None
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        max_contour = contour

# Create a mask of the hair region
mask = np.zeros_like(gray)
cv2.drawContours(mask, [max_contour], 0, (255, 255, 255), -1)

# Apply the mask to the original image to extract the hair region
hair = cv2.bitwise_and(img, img, mask=mask)

# Display the hair region
cv2.imshow('Hair Region', hair)
cv2.imwrite('hair.jpg', hair)
cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2
import numpy as np
from collections import Counter

# Load the image
img = cv2.imread('hair.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the color ranges for hair color detection
color_ranges = {'blackbrown': [(0, 0, 0), (180, 255, 63)],
                'brown': [(0, 40, 20), (180, 255, 255)],
                'blonde': [(20, 40, 20), (40, 255, 255)],
                'black': [(170, 40, 20), (180, 255, 255)]}

# Create an empty list to store the hair colors detected
hair_colors = []

# Loop over each color range and detect hair pixels
for color, (lower_bound, upper_bound) in color_ranges.items():
    # Threshold the image using the lower and upper bounds
    mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
    
    # Apply a median filter to remove noise from the mask
    mask = cv2.medianBlur(mask, 5)

    # Find the contours in the mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour and draw a bounding box around it
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the mean color of the hair region
        hair_color = cv2.mean(img[y:y+h, x:x+w], mask=mask[y:y+h, x:x+w])[:3]
        hair_colors.append(color)

        # Display the hair color name
        cv2.putText(img, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Count the hair colors detected and display the dominant hair color
dominant_color = Counter(hair_colors).most_common(1)[0][0]
cv2.putText(img, 'Dominant Color: {}'.format(dominant_color), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
print("hair colour is:", dominant_color)
# Display the image with hair color detection and bounding box
cv2.imshow('Hair Color Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
