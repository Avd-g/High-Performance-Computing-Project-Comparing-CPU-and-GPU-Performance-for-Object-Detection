# High-Performance-Computing-Project-Comparing-CPU-and-GPU-Performance-for-Object-Detection
### High-Performance Computing: CPU vs GPU Object Detection

This repository contains Python implementations and performance evaluations of object detection algorithms comparing traditional CPU-based Haar Cascade classifiers and GPU-optimized YOLOv5 deep learning models. The primary goal is to assess processing efficiency, accuracy, memory management, and hardware utilization across videos of different resolutions.

#### Project Overview:

* **Objective:** Evaluate and compare CPU and GPU performance for real-time object detection tasks.
* **Algorithms Compared:**

  * CPU-based Haar Cascade classifier.
  * GPU-based YOLOv5 deep learning model.

#### Implementation and Key Features:

* **CPU-based Detection:**

  * Utilizes OpenCV’s Haar Cascade classifiers.
  * Processes video frames sequentially, converting frames to grayscale for efficient detection.
  * Monitors memory usage and processing time for performance analysis.

* **GPU-based Detection:**

  * Implements YOLOv5 using PyTorch, optimized for GPU execution.
  * Batch processes frames to leverage parallel GPU computation capabilities.
  * Integrates comprehensive GPU monitoring using NVIDIA Management Library (NVML) for accurate hardware utilization metrics.

* **Performance Metrics:**

  * Measures total processing time, average time per frame, CPU and GPU memory usage, and GPU utilization rates.
  * Conducts thorough comparisons across different video resolutions to demonstrate practical application scenarios.

#### Key Findings:

* **Accuracy and Efficiency:** YOLOv5 significantly outperforms Haar Cascades, especially in high-resolution video streams, offering superior detection accuracy and processing speed.
* **Memory Management:** Efficient GPU memory management through explicit garbage collection and caching strategies ensures sustained high performance.

#### Pros and Cons Identified:

* **CPU (Haar Cascade):**

  * Pros: Low resource requirements, simple implementation, suitable for basic tasks.
  * Cons: Inefficient for complex or large-scale detection tasks, slower sequential processing.
* **GPU (YOLOv5):**

  * Pros: High-speed parallel processing, accurate detection in varied conditions, efficient handling of complex tasks.
  * Cons: High hardware costs, increased memory overhead for smaller tasks, dependency complexity.

#### Future Directions:

* Explore multi-threading and parallel processing optimizations for CPU code.
* Deployment of optimized YOLOv5 models on edge computing platforms like NVIDIA Jetson.
* Investigate lightweight YOLO models (e.g., YOLOv5s) for improved GPU resource utilization.

This project provides valuable insights and practical benchmarks for developers aiming to deploy efficient and accurate real-time object detection systems.
