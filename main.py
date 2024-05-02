from mpi4py import MPI
import cv2
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define tasks
tasks = ["grayscale", "edge_detection", "object_detection", "blur"]

# Video capture
cap = cv2.VideoCapture(0)  # Change '0' to the appropriate video source

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        break

    # Broadcast the frame to all processors
    frame = comm.bcast(frame, root=0)

    # Scatter tasks
    task = tasks[rank]

    # Perform tasks on the frame
    if task == "grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif task == "edge_detection":
        frame = cv2.Canny(frame, 100, 200)
    elif task == "object_detection":
        # Implement object detection code here
        pass
    elif task == "blur":
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Reduce the size of the frame
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Gather the processed frames
    processed_frames = comm.gather(frame, root=0)

    # Concatenate and display the processed frames (if rank == 0)
    if rank == 0:
        concatenated_frame = np.concatenate(processed_frames, axis=1)
        cv2.imshow("Consolidated Frame", concatenated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Finalize MPI
comm.Barrier()