from mpi4py import MPI
import cv2
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the target screen resolution
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

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
        # Process 3: Rotate the image by 90 degrees clockwise
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def main():
    cap = None
    if rank == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        frame = None
        if rank == 0:
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

        # Gather all frames at rank 4 process
        all_frames = comm.gather(processed_frame, root=4)

        # Rank 4 concatenates and displays frames
        if rank == 4:
            if all_frames[0] is not None:
                # Get the dimensions of the screen partition
                num_frames = len(all_frames)
                partition_width = SCREEN_WIDTH // num_frames
                partition_height = SCREEN_HEIGHT

                # Resize frames to fit the screen partition
                resized_frames = [cv2.resize(frame, (partition_width, partition_height)) for frame in all_frames]

                # Concatenate frames horizontally
                final_frame = cv2.hconcat(resized_frames)

                # Display the concatenated frames
                if final_frame is not None:
                    cv2.imshow('Video Processing MPI', final_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if rank == 0:
        cap.release()
    if rank == 4:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
