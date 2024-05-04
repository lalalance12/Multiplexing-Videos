import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import threading
import numpy as np

# Initialize capture from webcam
cap = cv2.VideoCapture(0)

# Define the desired size for each small frame
frame_width, frame_height = 320, 240

# Shared frame storage
frame_storage = {
    'original': None,
    'blur': None,
    'vibrance': None,
    'greyscale': None,
    'high_green_hue': None
}

def fetch_video():
    while True:
        ret, frame = cap.read()
        if ret:
            frame_storage['original'] = frame

def apply_blur():
    while True:
        if frame_storage['original'] is not None:
            frame_storage['blur'] = cv2.GaussianBlur(frame_storage['original'], (21, 21), 0)

def apply_vibrance():
    while True:
        if frame_storage['original'] is not None:
            frame = frame_storage['original']
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 100)  # increase saturation
            frame_storage['vibrance'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def apply_greyscale():
    while True:
        if frame_storage['original'] is not None:
            grey = cv2.cvtColor(frame_storage['original'], cv2.COLOR_BGR2GRAY)
            frame_storage['greyscale'] = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

def apply_high_green_hue():
    while True:
        if frame_storage['original'] is not None:
            frame = frame_storage['original']
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = 60  # set hue to green
            frame_storage['high_green_hue'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def update_display():
    # Ensure all frames are available and resized
    if all(frame is not None for frame in frame_storage.values()):
        resized_frames = {effect: cv2.resize(frame_storage[effect], (frame_width, frame_height)) for effect in frame_storage}
        
        # Row 1: Blur and Vibrance
        row1 = cv2.hconcat([resized_frames['blur'], resized_frames['vibrance']])
        # Row 2: Greyscale and High Green Hue
        row2 = cv2.hconcat([resized_frames['greyscale'], resized_frames['high_green_hue']])
        # Row 3: Original, centered
        original_resized = cv2.resize(frame_storage['original'], (2*frame_width, frame_height))  # Double width for centering
        row3 = original_resized

        # Combine all rows vertically
        combined_frame = cv2.vconcat([row1, row2, row3])
        
        # Convert to a format suitable for Tkinter
        img = Image.fromarray(combined_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
    lbl_video.after(10, update_display)

# Set up the GUI
root = tk.Tk()
root.title("Video Processing")
lbl_video = tk.Label(root)
lbl_video.pack()

# Start video fetch and processing threads
threads = [
    threading.Thread(target=fetch_video),
    threading.Thread(target=apply_blur),
    threading.Thread(target=apply_vibrance),
    threading.Thread(target=apply_greyscale),
    threading.Thread(target=apply_high_green_hue)
]

for thread in threads:
    thread.daemon = True
    thread.start()

# Start the video display in a separate thread
thread_display = threading.Thread(target=update_display)
thread_display.daemon = True
thread_display.start()

# Start the Tkinter main loop
root.mainloop()
