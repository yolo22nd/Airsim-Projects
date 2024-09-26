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

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)


from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox



model = torch.jit.load("best.torchscript")  # Load the TorchScript model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

img_size = 640


# Global variables
running = True
latest_image = None

def get_image():
    response = client.simGetImage("front", airsim.ImageType.Scene)
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))  # Resize image to 640x640
    return img_rgb

def display_image(image):
    """Display the processed image in a window."""
    if image is not None:
        cv2.imshow("Drone Camera", image)
        cv2.waitKey(0.5)  # Display for a brief moment


coco_classes = ['square', 'triangle', 'target', 'hotspot', 'aerfgra']


async def main():
    global running

    try:
        
        # Takeoff
        print("Taking off...")
        client.takeoffAsync()
        await asyncio.sleep(5)  # Wait for takeoff completion

        # Move to an initial altitude
        print("Moving to 7 meters altitude...")
        client.moveToPositionAsync(0, 0, -7, 5)
        await asyncio.sleep(5)  # Wait for 5 seconds

        start_time = time.time()
        # person_detected = False
        latest_image = get_image()  # Get the latest image from the drone

        pose = client.simGetVehiclePose()

        # Extract the position
        current_position = pose.position

        # Extract the coordinates (x, y, z)
        x = current_position.x_val
        y = current_position.y_val
        z = current_position.z_val


        while running and (time.time() - start_time < 30):
            # Get the latest image
            latest_image = get_image()  # Get the latest image from the drone

            if latest_image is not None:
                # Preprocess image for TorchScript model
                img = letterbox(latest_image, img_size)[0]  # Resize and pad the image to the target size
                img = img.transpose((2, 0, 1))[::-1]  # Convert from HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float() / 255.0  # Normalize pixel values (0 to 1)
                
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)  # Add batch dimension if required

                # Inference using TorchScript model
                with torch.no_grad():
                    pred = model(img)[0]  # Run the model

                # Apply Non-Maximum Suppression (NMS) and keep only "target" class (assuming class index 4)
                pred = non_max_suppression(pred, conf_thres=0.60, iou_thres=0.45, classes=[4], agnostic=False)

                target_boxes = []

                # Process predictions
                det = pred[0]  # Get predictions for the first image
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], latest_image.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        if conf > 0.60 and int(cls) == 3:  # Class 3 is "target"
                            label = f'target {conf:.2f}'
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            cv2.rectangle(latest_image, c1, c2, (0, 255, 0), thickness=3)
                            cv2.putText(latest_image, label, (c1[0], c1[1] - 2), 0, 0.8, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                            target_boxes.append(xyxy)

                if target_boxes:
                    target_detected = True
                    xyxy = target_boxes[0]  # Get the first detected target box
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2

                    pose = client.simGetVehiclePose()

                    # Extract the drone's current position
                    current_position = pose.position
                    x, y, z = current_position.x_val, current_position.y_val, current_position.z_val

                    # Adjust drone movement based on target's position
                    if center_x < img_size / 3:  # Move left if target is on the left
                        y -= 2
                    elif center_x > 2 * img_size / 3:  # Move right if target is on the right
                        y += 2

                    if center_y < img_size / 3:  # Move backward if target is at the bottom
                        x -= 2
                    elif center_y > 2 * img_size / 3:  # Move forward if target is at the top
                        x += 2

                    # Move the drone to the adjusted position
                    client.moveToPositionAsync(x, y, z, 3)  # Move slowly to avoid delayed feed issues
                    await asyncio.sleep(3)  # Wait for the drone to move

                else:
                    # No target detected, move forward
                    x += 10  # Move forward by 10 units
                    client.moveToPositionAsync(x, y, z, 6)  # Move forward at higher speed if no target
                    await asyncio.sleep(3)  # Wait for the drone to move

                # Display the processed image
                display_image(latest_image)


        pose = client.simGetVehiclePose()

        # Extract the position
        current_position = pose.position

        # Extract the coordinates (x, y, z)
        x = current_position.x_val
        y = current_position.y_val
        z = current_position.z_val


        # Return to launch

        print("lowering...")
        client.moveToPositionAsync(x, y, -1, 4)
        await asyncio.sleep(4)

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
    asyncio.run(main()) 