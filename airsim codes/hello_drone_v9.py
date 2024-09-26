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

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Load the TorchScript model
model = torch.jit.load("coco/best_b.torchscript") 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the desired device (CPU or GPU)
model.eval()  # Set the model to evaluation mode

img_size = 640

# Global variables
running = True
latest_image = None
processed_image = None
image_lock = threading.Lock()
processed_image_lock = threading.Lock()

coco_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", 
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
                "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", 
                "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", 
                "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
                "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", 
                "chair", "couch", "potted plant", "bed", "dining table", "toilet", "TV", "laptop", 
                "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
                "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair dryer", 
                "toothbrush"]

def get_image():
    responses = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))  # Resize image to 640x640
    return img_rgb

def update_image():
    global latest_image, running
    try:
        while running:
            new_image = get_image()
            with image_lock:
                latest_image = new_image
            time.sleep(0.1)  # Adjust sleep time as necessary
    except Exception as e:
        logger.error(f"Error in update_image thread: {e}")
        running = False

def process_image():
    global latest_image, processed_image, running
    try:
        while running:
            with image_lock:
                if latest_image is not None:
                    # Preprocess image
                    img = cv2.resize(latest_image, (640, 640))
                    img = torch.from_numpy(img).to(device)  # Ensure the image is on the correct device
                    img = img.float() / 255.0
                    img = img.permute(2, 0, 1).unsqueeze(0)  # Change to CHW format

                    # Inference using the TorchScript model
                    with torch.no_grad():  # Disable gradient calculation for inference
                        results = model(img)  # Call the TorchScript model directly

                    # Process predictions
                    boxes = results[0].boxes.xyxy.cpu().numpy()  # Ensure box coordinates are on CPU
                    confs = results[0].boxes.conf.cpu().numpy()  # Ensure confidence is on CPU
                    cls = results[0].boxes.cls.cpu().numpy().astype(int)  # Ensure class index is on CPU

                    if boxes is not None and len(boxes) > 0:
                        for i in range(len(boxes)):
                            xyxy = boxes[i]
                            conf = confs[i]
                            class_id = cls[i]

                            # Draw bounding box
                            label = f'{coco_classes[class_id]} {conf:.2f}'
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            cv2.rectangle(latest_image, c1, c2, (0, 255, 0), thickness=3)
                            cv2.putText(latest_image, label, (c1[0], c1[1] - 2), 0, 0.8, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                    with processed_image_lock:
                        processed_image = latest_image

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

        # Takeoff
        print("Taking off...")
        client.takeoffAsync()
        await asyncio.sleep(5)  # Allow time for takeoff

        # Move to 15 meters height
        print("Moving to 15 meters altitude...")
        client.moveToPositionAsync(0, 0, -15, 5)
        await asyncio.sleep(5)

        # Additional flight commands...

    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        running = False
        update_thread.join()
        processing_thread.join()
        display_thread.join()
        cv2.destroyAllWindows()
        client.armDisarm(False)
        client.enableApiControl(False)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
    