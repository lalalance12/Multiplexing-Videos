# Multiplexing Videos Using MPI Video Processing 

## Overview

This Python script demonstrates a parallel video processing application using MPI (Message Passing Interface). It captures video from a camera, applies different processing effects to the frames using multiple processes, and displays the results in real-time.

## Dependencies

- mpi4py
- OpenCV (cv2)
- NumPy

## MPI Setup

The script uses MPI to distribute the workload across multiple processes:

- Process 0: Captures video frames
- Process 1: Applies Gaussian blur
- Process 2: Converts frames to grayscale
- Process 3: Rotates frames by 90 degrees clockwise
- Process 4: Gathers processed frames and displays the result

## Main Components

### 1. Initialization

```python
from mpi4py import MPI
import cv2
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```

This section imports necessary libraries and initializes MPI communication.

### 2. Frame Processing

```python
def process_frame(frame, rank):
    # Process frames based on rank
    # ...
```

This function applies different processing effects to the frame based on the process rank.

### 3. Main Loop

```python
def main():
    # ...
```

The main function contains the primary video processing loop:

1. Process 0 captures video frames
2. Frames are broadcast to other processes
3. Each process applies its specific effect
4. Processed frames are gathered at Process 4
5. Process 4 concatenates and displays the frames

## Screen Resolution

The script targets a specific screen resolution:

```python
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
```

## Usage

To run the script, use the `mpiexec` command with at least 5 processes:

```
mpiexec -n 5 python script_name.py
```

## Notes

- The script uses `cv2.CAP_DSHOW` for video capture, which is specific to Windows. For other operating systems, this parameter may need to be adjusted.
- The script continues running until the 'q' key is pressed while the output window is in focus.
- Ensure that a camera is connected and accessible to the system running the script.

## Potential Improvements

1. Error handling for cases where a camera is not available
2. Command-line arguments for customizing screen resolution and number of processes
3. Additional video processing effects
4. Performance optimizations for real-time processing
