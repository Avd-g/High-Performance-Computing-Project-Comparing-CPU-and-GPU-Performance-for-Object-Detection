import time
import cv2
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert bytes to MB

def process_entire_video_cpu(video_src, cascade_src):
    # Initialize video capture and Haar cascade
    cap = cv2.VideoCapture(video_src)
    car_cascade = cv2.CascadeClassifier(cascade_src)

    initial_memory = get_memory_usage()
    start_time = time.time()

    frame_count = 0  
    while True:
        ret, img = cap.read()
        if not ret:  # Stop if no frames are left
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

       
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        # Draw rectangles on detected cars 
        for (x, y, w, h) in cars:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Car Detection', img)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # Increment frame count

    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()

    # Measure memory after processing
    final_memory = get_memory_usage()

    # Output performance information
    print(f"Total Frames Processed: {frame_count}")
    print(f"CPU Processing Time for Entire Video: {end_time - start_time:.2f} seconds")
    print(f"Average Time Per Frame: {(end_time - start_time) / frame_count:.4f} seconds")
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")
    print(f"Final Memory Usage: {final_memory:.2f} MB")
    print(f"Memory Usage Difference: {final_memory - initial_memory:.2f} MB")


video_src = ""  # Path to the video file
cascade_src = ""  # Path to the Haar cascade XML file

# Run the function
process_entire_video_cpu(video_src, cascade_src)

