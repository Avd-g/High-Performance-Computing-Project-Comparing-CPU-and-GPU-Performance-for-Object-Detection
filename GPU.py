import torch
import cv2
import psutil
import os
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates

# NVML for GPU monitoring
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)  # single GPU 

# Functions for monitoring
def get_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # MB

def get_gpu_memory_usage():
    mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    return mem_info.used / (1024 * 1024)  # MB

def get_gpu_utilization():
    util_rates = nvmlDeviceGetUtilizationRates(gpu_handle)
    return util_rates.gpu  # %

# Load YOLOv5 model to GPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True, force_reload=True)
model.to('cuda')  # Move model to GPU

target_class = "car"


total_frames = 0
total_processing_time = 0
gpu_utilization_data = []

initial_cpu_memory = get_cpu_memory_usage()
initial_gpu_memory = get_gpu_memory_usage()

# Function to detect objects in batches
def detect_objects_batch(frames):
    # Convert each frame to RGB
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    # Run inference on batch
    with torch.no_grad():  
        results = model(frames_rgb)
        torch.cuda.synchronize()  # Ensure GPU finishes computation
    # Return detections as a list of DataFrames
    detections_batch = results.pandas().xyxy
    return detections_batch

# Process video with batching
def process_video(video_path, batch_size=16):
    global total_frames, total_processing_time
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video source")
        return

    print("Press 'q' to exit...")
    frame_batch = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to reduce memory usage
        frame = cv2.resize(frame, (640, 640))

        frame_batch.append(frame)
        if len(frame_batch) == batch_size:
            start_time = time.time()
            detections_batch = detect_objects_batch(frame_batch)
            end_time = time.time()

            total_frames += len(frame_batch)
            total_processing_time += (end_time - start_time)
            gpu_utilization_data.append(get_gpu_utilization())

            # Process detections for each frame
            for frame, detections in zip(frame_batch, detections_batch):
                for _, row in detections[detections['name'] == target_class].iterrows():
                    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                    confidence = row['confidence']
                    label = f"{target_class}: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.imshow("YOLOv5 Car Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            
            for frame, detections in zip(frame_batch, detections_batch):
                del frame
                del detections
            del frame_batch
            del detections_batch
            torch.cuda.empty_cache()
            gc.collect()  # Trigger garbage collection

            frame_batch = []

    cap.release()
    cv2.destroyAllWindows()

# video path
video_path = ""
process_video(video_path)

# Final performance metrics
final_cpu_memory = get_cpu_memory_usage()
final_gpu_memory = get_gpu_memory_usage()

average_time_per_frame = total_processing_time / total_frames if total_frames > 0 else 0
average_gpu_utilization = sum(gpu_utilization_data) / len(gpu_utilization_data) if gpu_utilization_data else 0

# Print performance metrics
print("\n===== Performance Metrics =====")
print(f"Total Frames Processed: {total_frames}")
print(f"GPU Processing Time for Entire Video: {total_processing_time:.4f} seconds")
print(f"Average Time Per Frame: {average_time_per_frame:.4f} seconds")
print(f"Initial CPU Memory Usage: {initial_cpu_memory:.2f} MB")
print(f"Final CPU Memory Usage: {final_cpu_memory:.2f} MB")
print(f"CPU Memory Usage Difference: {final_cpu_memory - initial_cpu_memory:.2f} MB")
print(f"Initial GPU Memory Usage: {initial_gpu_memory:.2f} MB")
print(f"Final GPU Memory Usage: {final_gpu_memory:.2f} MB")
print(f"GPU Memory Usage Difference: {final_gpu_memory - initial_gpu_memory:.2f} MB")
print(f"Average GPU Utilization: {average_gpu_utilization:.2f}%")

