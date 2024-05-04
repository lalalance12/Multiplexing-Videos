from mpi4py import MPI
import cv2
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def process_frame(frame, rank):
    # Process frames based on rank; Process 0 does nothing to the frame
    if rank == 1:
        # Process 1: Apply blur
        return cv2.GaussianBlur(frame, (21, 21), 0)
    elif rank == 2:
        # Process 2: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif rank == 3:
        # Process 3: Apply high vibrance
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 100)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif rank == 4:
        # Process 4: Apply high green hue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = 60  # Set hue to green
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame  # If rank 0 or any other rank, return the frame as is

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Broadcast the frame to all processes except the root (rank 0 does nothing to the frame)
        if rank == 0:
            for i in range(1, size):
                comm.send(frame, dest=i)
        else:
            frame = comm.recv(source=0)

        # Process the frame except for rank 0
        processed_frame = process_frame(frame, rank) if rank != 0 else frame

        # Gather all frames at root process
        all_frames = comm.gather(processed_frame, root=0)

        # Concatenate and display frames in the root process
        if rank == 0:
            if all_frames[0] is not None:  # all_frames[0] is the original frame from rank 0
                all_frames = all_frames[1:]  # Ignore the unprocessed frame from rank 0
            final_frame = cv2.hconcat(all_frames) if all_frames else None
            if final_frame is not None:
                cv2.imshow('Video Processing MPI', final_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
