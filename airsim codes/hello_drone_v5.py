# basic torch

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

# Add the path to the yolov5 repository
yolov5_path = Path('c:/Users/omtan/OneDrive/Desktop/Phoenix/airsim codes/yolov5')
sys.path.append(str(yolov5_path))

# Import YOLOv5 utils and models explicitly from yolov5 directory
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.plots import save_one_box
from yolov5.utils.augmentations import letterbox

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)



#load yolov5 model
weights_path = "C:\\Users\\omtan\\OneDrive\\Desktop\\Phoenix\\Airsim-Projects\\airsim codes\\yolov5\\best_ak.pt"  # Note: Changed to .pt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640


# Global variables
running = True
latest_image = None
processed_image = None
image_lock = threading.Lock()
processed_image_lock = threading.Lock()

# Ensure 'names' is defined for your YOLO model classes
# names = ['square', 'triangle', 'circle', 'hotspot', 'target']

def get_image():
    responses = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def update_image():
    global latest_image, running
    try:
        while running:
            new_image = get_image()
            with image_lock:
                latest_image = new_image
            time.sleep(0.033)  # Approximately 30 FPS
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
                    img = letterbox(latest_image, img_size, stride=stride)[0]
                    img = img.transpose((2, 0, 1))[::-1]
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(device)
                    img = img.float()
                    img /= 255.0
                    if len(img.shape) == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    pred = model(img)

                    # Apply NMS
                    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

                    # Process predictions
                    det = pred[0]
                    if len(det):
                        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], latest_image.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{names[int(cls)]} {conf:.2f}'
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            cv2.rectangle(latest_image, c1, c2, (0, 255, 0), thickness=3)
                            cv2.putText(latest_image, label, (c1[0], c1[1] - 2), 0, 0.8, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                    with processed_image_lock:
                        processed_image = latest_image

                    frame_count += 1
                    if frame_count % 30 == 0:
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

async def main():
    global running
    try:
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

        # Descend with yaw
        print("Descending with yaw...")
        client.moveToPositionAsync(0, 0, -15, 5)
        client.rotateToYawAsync(360, 10)
        await asyncio.sleep(10)

        # Move to 1 meters height
        print("Moving to 1 meter altitude...")
        client.moveToPositionAsync(0, 0, -1, 5)
        await asyncio.sleep(5)  # Allow time to reach the target altitude

        # Land
        print("Landing...")
        client.landAsync()
        
        # Wait for the drone to actually land
        while client.getMultirotorState().landed_state != airsim.LandedState.Landed:
            await asyncio.sleep(1)
        
        print("Landing complete")

    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
    finally:
        graceful_shutdown()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
