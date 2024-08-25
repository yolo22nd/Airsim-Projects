import airsim
import cv2
import numpy as np
import threading
import time
import torch
from pathlib import Path
import sys
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the path to the yolov10 repository
yolov10_path = Path('c:/Users/omtan/OneDrive/Desktop/Phoenix/Airsim-Projects/airsim codes/yolov10')
sys.path.append(str(yolov10_path))

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Load YOLOv10 model
from ultralytics import YOLO
import logging


logging.getLogger("ultralytics").setLevel(logging.ERROR)
# model = YOLO("coco/yolov10s.pt")
model = YOLO("coco/yolov10b.pt")
# model = YOLO("coco/yolov10x.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the desired device (CPU or GPU)
img_size = 640

# Global variables
running = True
latest_image = None
processed_image = None
image_lock = threading.Lock()
processed_image_lock = threading.Lock()
FRAME_RATE = 22
FRAME_INTERVAL = 1.0 / FRAME_RATE

coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck","boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench","bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra","giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee","skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove","skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup","fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch","potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse","remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink","refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier","toothbrush"]

def get_image():
    responses = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))  # Resize image to 640x640
    return img_rgb

last_frame_time = time.time()
target_interval = 1.0 / 30  # Start with targeting 30 FPS

def update_image():
    global latest_image, running, last_frame_time, target_interval
    try:
        while running:
            new_image = get_image()
            with image_lock:
                latest_image = new_image
            
            current_time = time.time()
            processing_time = current_time - last_frame_time
            sleep_time = max(0, target_interval - processing_time)
            time.sleep(sleep_time)
            
            # Adjust target interval based on actual FPS
            actual_interval = time.time() - last_frame_time
            target_interval = target_interval * 0.9 + actual_interval * 0.1  # Smooth adjustment
            target_interval = max(1.0/30, min(1.0/15, target_interval))  # Keep between 15 and 30 FPS
            
            last_frame_time = time.time()
    except Exception as e:
        logger.error(f"Error in update_image thread: {e}")
        running = False

def process_image():
    global latest_image, processed_image, running
    frame_count = 0
    start_time = time.time()
    try:
        while running:
            with image_lock:
                if latest_image is not None:
                    # Preprocess image
                    img = cv2.resize(latest_image, (640, 640))
                    img = torch.from_numpy(img).to(device)
                    img = img.float() / 255.0
                    img = img.permute(2, 0, 1).unsqueeze(0)

                    # Inference
                    results = model.predict(img, conf=0.25)

                    # Process predictions
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                xyxy = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())

                                # Draw bounding box
                                label = f'{coco_classes[cls]} {conf:.2f}'
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                cv2.rectangle(latest_image, c1, c2, (0, 255, 0), thickness=3)
                                cv2.putText(latest_image, label, (c1[0], c1[1] - 2), 0, 0.8, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                    with processed_image_lock:
                        processed_image = latest_image

                    frame_count += 1
                    if frame_count % 30 == 0:  # Display FPS every 5 seconds (assuming 6 FPS)
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        logger.info(f"Processing speed: {fps:.2f} FPS")
                        frame_count = 0
                        start_time = time.time()
            time.sleep(0.001)
    except Exception as e:
        logger.error(f"Error in process_image thread: {e}")
        running = False



def display_image():
    global processed_image, running
    try:
        while running:
            with processed_image_lock:
                if processed_image is not None:
                    cv2.imshow("Drone Camera", processed_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                running = False
                break
    except Exception as e:
        logger.error(f"Error in display_image thread: {e}")
        running = False
    finally:
        cv2.destroyAllWindows()



def warm_up_model(num_warmup_iterations=5):
    logger.info("Warming up the model...")
    dummy_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)  # Create a dummy image
    for _ in range(num_warmup_iterations):
        img = cv2.resize(dummy_image, (640, 640))
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        # Run inference
        _ = model.predict(img, conf=0.25)  # Discard results

    logger.info("Model warm-up complete.")

async def main():
    global running
    try:
        # Warm up the model before takeoff
        warm_up_model()

        # Start threads
        update_thread = threading.Thread(target=update_image)
        processing_thread = threading.Thread(target=process_image)
        display_thread = threading.Thread(target=display_image)

        update_thread.start()
        processing_thread.start()
        display_thread.start()

        def graceful_shutdown():
            global running
            logger.info("Initiating graceful shutdown...")
            running = False
            update_thread.join()
            processing_thread.join()
            display_thread.join()
            cv2.destroyAllWindows()
            client.armDisarm(False)
            client.enableApiControl(False)
            logger.info("Shutdown complete")

        # Takeoff
        print("Taking off...")
        client.takeoffAsync()
        await asyncio.sleep(5)  # Allow time for takeoff

        # Move to 15 meters height
        print("Moving to 15 meters altitude...")
        client.moveToPositionAsync(0, 0, -15, 5)
        await asyncio.sleep(5)

        # Square pattern
        print("Starting square pattern...")
        client.moveToPositionAsync(10, 0, -15, 5)
        await asyncio.sleep(5)
        client.moveToPositionAsync(10, 10, -15, 5)
        await asyncio.sleep(5)
        client.moveToPositionAsync(0, 10, -15, 5)
        await asyncio.sleep(5)
        client.moveToPositionAsync(0, 0, -15, 5)
        await asyncio.sleep(5)

        # Spiral ascent
        print("Starting spiral ascent...")
        for i in range(5):
            client.moveToPositionAsync(5*np.cos(i*np.pi/2), 5*np.sin(i*np.pi/2), -20-i*2, 5)
            await asyncio.sleep(5)

        # Return to launch
        print("Returning to launch...")
        client.moveToPositionAsync(0, 0, -15, 5)
        await asyncio.sleep(5)
        client.landAsync()
        await asyncio.sleep(5)

        logger.info("Mission completed")
        graceful_shutdown()
    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        graceful_shutdown()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
