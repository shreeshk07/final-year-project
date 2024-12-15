##import cv2
##import numpy as np
### load Haar cascade classifier for eyes detection
##eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
##
### initialize the video capture object
##cap = cv2.VideoCapture(0)
##
### loop over frames from the video stream
##while True:
##    # read the frame from the video stream
##    ret, frame = cap.read()
##
##    # convert the frame to grayscale
##    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
##
##    # detect eyes in the grayscale frame
##    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
##
##    # loop over the detected eyes and draw rectangles around them
##    for (x, y, w, h) in eyes:
##        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
##
##    # show the output frame
##    cv2.imshow('Eye Detection', frame)
##    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
##
##    # Define color ranges for different eye colors in HSV color space
##    color_ranges = {
##        'blue': [(100, 50, 50), (130, 255, 255)],
##        'black': [(30, 50, 50), (80, 255, 255)],
##        'brown': [(0, 50, 50), (30, 255, 255)]
##    }
##
##    # Threshold the image based on the defined color ranges
##    eye_color = None
##    for color, (lower, upper) in color_ranges.items():
##        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
##        count = cv2.countNonZero(mask)
##        if count > 0:
##            eye_color = color
##            break
##
##    # Display the detected eye color
##    if eye_color:
##        print("The detected eye color is: ", eye_color)
##    else:
##        print("No eye color detected.")
##
##    # Convert the image to HSV color space
##    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
##
##    # Define the lower and upper bounds for hair color in HSV
##    lower_bound = np.array([0, 0, 0])
##    upper_bound = np.array([255, 255, 70])
##
##    # Create a mask for hair color using the lower and upper bounds
##    mask = cv2.inRange(hsv, lower_bound, upper_bound)
##
##    # Apply the mask to the original image
##    result = cv2.bitwise_and(frame, frame, mask=mask)
##
##    # Convert the result image to grayscale
##    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
##
##    # Apply a threshold to the grayscale image to make the hair stand out
##    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
##
##    # Find contours in the thresholded image
##    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##
##    # Draw rectangles around the detected hair regions
##    for contour in contours:
##        x, y, w, h = cv2.boundingRect(contour)
##        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
##
##    # Show the result
##    cv2.imshow('Hair Detection', frame)
##    
##    from collections import Counter
##
##    
##    # Convert the image to HSV color space
##    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
##
##    # Define the color ranges for hair color detection
##    color_ranges = {'blackbrown': [(0, 0, 0), (180, 255, 63)],
##                    'brown': [(0, 40, 20), (180, 255, 255)],
##                    'blonde': [(20, 40, 20), (40, 255, 255)],
##                    'black': [(170, 40, 20), (180, 255, 255)]}
##
##    # Create an empty list to store the hair colors detected
##    hair_colors = []
##
##    # Loop over each color range and detect hair pixels
##    for color, (lower_bound, upper_bound) in color_ranges.items():
##        # Threshold the image using the lower and upper bounds
##        mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
##        
##        # Apply a median filter to remove noise from the mask
##        mask = cv2.medianBlur(mask, 5)
##
##        # Find the contours in the mask
##        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##
##        # Loop over each contour and draw a bounding box around it
##        for contour in contours:
##            (x, y, w, h) = cv2.boundingRect(contour)
##            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
##
##            # Get the mean color of the hair region
##            hair_color = cv2.mean(frame[y:y+h, x:x+w], mask=mask[y:y+h, x:x+w])[:3]
##            hair_colors.append(color)
##
##            
##    dominant_color = Counter(hair_colors).most_common(1)[0][0]
##
##    print("hair colour is:", dominant_color)
##
##
##    # exit the loop if the 'q' key is pressed
##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        break
##
### release the video capture object and close all windows
##cap.release()
##cv2.destroyAllWindows()
##
##
##


from tkinter import *
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image
import pickle
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
#import mediapipe as mp

#from utils import CvFpsCalc
from gtts import gTTS
from playsound import  playsound




def start():
    import cv2
    import numpy as np
    # load Haar cascade classifier for eyes detection
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    # initialize the video capture object
    cap = cv2.VideoCapture(0)

    # loop over frames from the video stream
    while True:
        # read the frame from the video stream
        ret, frame = cap.read()

        # convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect eyes in the grayscale frame
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # loop over the detected eyes and draw rectangles around them
        for (x, y, w, h) in eyes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # show the output frame
        cv2.imshow('Eye Detection', frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define color ranges for different eye colors in HSV color space
        color_ranges = {
            'blue': [(100, 50, 50), (130, 255, 255)],
            'black': [(30, 50, 50), (80, 255, 255)],
            'brown': [(0, 50, 50), (30, 255, 255)],
            'green': [(10, 50, 50), (20, 255, 255)],
            'gray': [(20, 50, 50), (40, 255, 255)],
            'red': [(40, 50, 50), (60, 255, 255)]
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
        #messagebox.showinfo('eye colour is: ',eye_color)   

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for hair color in HSV
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([255, 255, 70])

        # Create a mask for hair color using the lower and upper bounds
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Apply the mask to the original image
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert the result image to grayscale
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to the grayscale image to make the hair stand out
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around the detected hair regions
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show the result
        cv2.imshow('Hair Detection', frame)
        
        from collections import Counter

        
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the color ranges for hair color detection
        color_ranges = {'blackbrown': [(0, 0, 0), (180, 255, 63)],
                        'brown': [(0, 40, 20), (180, 255, 255)],
                        'blonde': [(20, 40, 20), (40, 255, 255)],
                        'black': [(170, 40, 20), (180, 255, 255)],
                        'gray': [(180, 40, 20), (190, 255, 255)],
                        'red': [(190, 40, 20), (60, 255, 255)]}

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get the mean color of the hair region
                hair_color = cv2.mean(frame[y:y+h, x:x+w], mask=mask[y:y+h, x:x+w])[:3]
                hair_colors.append(color)

                
        dominant_color = Counter(hair_colors).most_common(1)[0][0]

        print("hair colour is:", dominant_color)
        #messagebox.showinfo('hair colour is: ',dominant_color)


        # exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()




    
               
              
master = Tk()
master.title('EYE AND HAIR COLOUR DETECTION')
master.geometry('850x750')
master.config(bg='black')

##img =Image.open("hand1.jpg")
##bg = ImageTk.PhotoImage(img)
##label = Label(master, image=bg)
##label.place(x = 0,y = 0)



w = tk.Label(master, 
		 text=" EYE AND HAIR COLOUR DETECTION ",
		 fg = "white",
		 bg = "black",
		 font = "Helvetica 20 bold italic")
w.pack()
w.place(x=450, y=0)


##T1 = Entry(master, width=20)
##T1.place(x=750, y=250, width=350, height=450)

##def delete():
##    T1.delete(0)

quitButton = Button(master,command=master.destroy, text="quit",fg="blue",bg="tan",width=18)
quitButton.place(x=650, y=140)
quitButton = Button(master,command=start,text="start",fg="blue",bg="tan",width=16)
quitButton.place(x=500, y=140)
##quitButton = Button(master, text = "Delete", command = delete,fg="blue",bg="tan",width=16)
##quitButton.place(x=800, y=140)
#quitButton.pack(pady = 5)

mainloop()
    
