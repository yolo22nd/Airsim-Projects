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

# Suppress output from the ultralytics library
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Add the path to the YOLOv10 repository
yolov10_path = Path('c:/Users/omtan/OneDrive/Desktop/Phoenix/Airsim-Projects/airsim codes/yolov10')
sys.path.append(str(yolov10_path))

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Load the new model that detects only one class (person)
from ultralytics import YOLO
model = YOLO("person_om.pt")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

img_size = 640

# Global variables
running = True
latest_image = None
processed_image = None
image_lock = threading.Lock()
processed_image_lock = threading.Lock()

coco_classes = ["person"]

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

                                # Draw bounding box for detected person
                                label = f'{coco_classes[cls]} {conf:.2f}'  # Only one class
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                cv2.rectangle(latest_image, c1, c2, (0, 255, 0), thickness=3)
                                cv2.putText(latest_image, label, (c1[0], c1[1] - 2), 0, 0.8, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                    with processed_image_lock:
                        processed_image = latest_image

            time.sleep(0.001)
    except Exception as e:
        logger.error(f"Error in process_image thread: {e}")
        running = False

def display_image(latest_image):
    """Display the processed image in a window."""
    try:
        if latest_image is not None:
            cv2.imshow("Drone Camera", latest_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return  # Exit if 'q' is pressed
    except Exception as e:
        logger.error(f"Error displaying image: {e}")


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
        warm_up_model()

        # Takeoff
        print("Taking off...")
        client.takeoffAsync()

        # Move to an initial altitude
        print("Moving to 3 meters altitude...")
        client.moveToPositionAsync(-3, 0, -2, 5)  # Move to (-1, 0, -2) for 3m altitude
        await asyncio.sleep(5)  # Wait for 5 seconds

        start_time = time.time()
        person_detected = False

        while running and (time.time() - start_time < 30):
            # Get the latest image
            latest_image = get_image()  # Get the latest image from the drone
            if latest_image is not None:
                # Preprocess image
                img = cv2.resize(latest_image, (640, 640))
                img = torch.from_numpy(img).to(device)
                img = img.float() / 255.0
                img = img.permute(2, 0, 1).unsqueeze(0)

                # Inference
                results = model.predict(img, conf=0.25)
                person_boxes = []

                # Process predictions
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())

                            if coco_classes[cls] == "person":
                                # Draw bounding box for detected person
                                label = f'{coco_classes[cls]} {conf:.2f}'
                                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                cv2.rectangle(latest_image, c1, c2, (0, 255, 0), thickness=3)
                                cv2.putText(latest_image, label, (c1[0], c1[1] - 2), 0, 0.8, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                                person_boxes.append(xyxy)

                if person_boxes:
                    person_detected = True
                    xyxy = person_boxes[0]
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    box_width = xyxy[2] - xyxy[0]

                    # Calculate target velocities for the drone
                    vx, vy, vz = 0, 0, 0  # Initialize velocities

                    # Determine horizontal movement based on horizontal position
                    if center_x < img_size / 3:  # Left third
                        print("Moving left")
                        vy = -1  # Move left
                    elif center_x > 2 * img_size / 3:  # Right third
                        print("Moving right")
                        vy = 1  # Move right
                    else:
                        print("Centering person in the middle")

                    # Determine distance adjustment based on bounding box size
                    # if box_width < img_size / 4:  # Small bounding box
                    #     print("Moving forward to close the distance")
                    #     vx = 1  # Move forward
                    # elif box_width > img_size / 2:  # Large bounding box
                    #     print("Moving backward to increase the distance")
                    #     vx = -1  # Move backward

                    # Determine vertical adjustment based on vertical position
                    if center_y < img_size / 3:  # Lower third
                        print("Ascending and move back to keep the person centered")
                        vz = -1  # Ascend
                        vx = -1  # Move backward

                    elif center_y > 2 * img_size / 3:  # Upper third
                        print("Descending and move forward to keep the person centered")
                        vz = 1  # Descend
                        vx = 1  # Move forward

                    # Move the drone based on calculated velocities
                    print(f"Moving with velocity (vx: {vx}, vy: {vy}, vz: {vz})")
                    client.moveByVelocityAsync(vx, vy, vz, 2)  # Move for 2 seconds
                    await asyncio.sleep(2)  # Allow time for the drone to move

                    display_image(latest_image)

                else:
                    await asyncio.sleep(0.1)  # Delay for next iteration if no person is detected

        # Return to launch
        print("Returning to launch...")
        client.moveToPositionAsync(0, 0, -5, 5)
        await asyncio.sleep(5)

        # Land the drone
        print("Landing...")
        client.landAsync()
        await asyncio.sleep(5)

    except Exception as e:
        logger.error(f"Error in main function: {e}")
    finally:
        running = False
        cv2.destroyAllWindows()
        client.armDisarm(False)
        client.enableApiControl(False)

if __name__ == "__main__":
    asyncio.run(main())  # Use asyncio.run to manage the event loop