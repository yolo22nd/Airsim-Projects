import cv2
import torch
import numpy as np
import time
import psutil
import GPUtil

# Load YOLOv5 model
# weights_path = "C:\\Users\\omtan\\OneDrive\\Desktop\\Phoenix\\airsim codes\\yolov5\\best.torchscript"
# weights_path = "C:\\Users\\omtan\\OneDrive\\Desktop\\Phoenix\\airsim codes\\yolov5\\best_ak.torchscript"
# weights_path = "coco/best_v5_x.torchscript"
weights_path = "coco/best.torchscript"
# weights_path = "best_ak.torchscript"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.jit.load(weights_path).to(device)
img_size = 640

# Create sample images with simulated detections
def generate_sample_image(num_objects=3):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(num_objects):
        color = tuple(np.random.randint(0, 256, 3).tolist())
        x1, y1 = np.random.randint(0, 600), np.random.randint(0, 440)
        x2, y2 = x1 + np.random.randint(20, 40), y1 + np.random.randint(20, 40)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img

# Preprocess image
def preprocess_image(img):
    img = cv2.resize(img, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

# Warm-up period
print("Warming up...")
for _ in range(50):
    img = generate_sample_image()
    img = preprocess_image(img)
    with torch.no_grad():
        _ = model(img)

# Benchmark function
def run_benchmark(num_frames, num_objects):
    print(f"Starting benchmark with {num_objects} simulated objects per frame...")
    frame_count = 0
    start_time = time.time()

    for _ in range(num_frames):
        img = generate_sample_image(num_objects)
        img = preprocess_image(img)

        with torch.no_grad():
            _ = model(img)

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            print(f"Processed {frame_count} frames. Current speed: {fps:.2f} FPS")

            cpu_percent = psutil.cpu_percent()
            mem_percent = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                gpu = GPUtil.getGPUs()[0]
                gpu_load = gpu.load * 100
                gpu_memory = gpu.memoryUtil * 100
                print(f"CPU: {cpu_percent:.1f}%, RAM: {mem_percent:.1f}%, GPU Load: {gpu_load:.1f}%, GPU Memory: {gpu_memory:.1f}%")

    total_time = time.time() - start_time
    average_fps = num_frames / total_time
    print(f"\nBenchmark complete. Average processing speed: {average_fps:.2f} FPS")
    return average_fps

# Run benchmarks with different numbers of objects
num_frames = 1000
object_counts = [0, 1, 3, 5, 10]
results = []

for count in object_counts:
    fps = run_benchmark(num_frames, count)
    results.append((count, fps))

# Print summary
print("\nBenchmark Summary:")
for count, fps in results:
    print(f"Objects per frame: {count}, Average FPS: {fps:.2f}")