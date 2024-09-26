import airsim
import cv2
import numpy as np
import time
import torch
import logging
import asyncio
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.augmentations import letterbox

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Load model (ensure this path is correct)
model = torch.jit.load("best.torchscript")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

img_size = 640
names = ['square', 'triangle', 'target', 'hotspot', 'aerfgra']  # Adjust this based on your model's classes

def get_image():
    responses = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))  # Resize image to 640x640
    return img_rgb

def is_target_centered(center_x, center_y, threshold=50):
    return abs(center_x - img_size/2) < threshold and abs(center_y - img_size/2) < threshold

async def main():
    running = True
    max_duration = 60  # Maximum run time in seconds (increased to 5 minutes)
    start_time = time.time()

    try:
        # Takeoff
        print("Taking off...")
        client.takeoffAsync()
        await asyncio.sleep(5)  # Wait for takeoff completion

        # Move to an initial altitude
        print("Moving to 7 meters altitude...")
        client.moveToPositionAsync(0, 0, -7, 5)
        await asyncio.sleep(5)  # Wait for 5 seconds

        while running and (time.time() - start_time < max_duration):
            latest_image = get_image()

            if latest_image is not None:
                # Preprocess image
                img = letterbox(latest_image, img_size)[0]
                img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float()
                img /= 255.0
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)

                # Inference
                with torch.no_grad():
                    pred = model(img)[0]

                # Apply NMS
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

                # Process predictions
                det = pred[0]
                target_detected = False
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], latest_image.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        cv2.rectangle(latest_image, c1, c2, (0, 255, 0), thickness=3)
                        cv2.putText(latest_image, label, (c1[0], c1[1] - 2), 0, 0.8, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                        if names[int(cls)] == 'hotspot':
                            target_detected = True
                            center_x = (c1[0] + c2[0]) / 2
                            center_y = (c1[1] + c2[1]) / 2

                            # Horizontal adjustment based on screen partition
                            hor_part_size = img_size / 9
                            if center_x < hor_part_size * 1 or center_x > hor_part_size * 8:
                                # Far left or right, move with 1 m/s
                                client.moveByVelocityAsync(0, -2.0 if center_x < hor_part_size * 1 else 1.0, 0, 1)
                                await asyncio.sleep(1)
                            elif center_x < hor_part_size * 3 or center_x > hor_part_size * 7:
                                # Near the center, but not aligned, move with 0.5 m/s
                                client.moveByVelocityAsync(0, -0.5 if center_x < hor_part_size * 3 else 0.5, 0, 1)
                                await asyncio.sleep(1)
                            # If it's in the 5th partition, it's centered horizontally.

                            # Vertical adjustment based on screen partition
                            ver_part_size = img_size / 9
                            if center_y < ver_part_size * 1 or center_y > ver_part_size * 8:
                                # Far up or down, move with 1 m/s
                                client.moveByVelocityAsync(-2.0 if center_y < ver_part_size * 1 else 1.0, 0, 0, 1)
                                await asyncio.sleep(1)
                            elif center_y < ver_part_size * 3 or center_y > ver_part_size * 7:
                                # Near the center, but not aligned, move with 0.5 m/s
                                client.moveByVelocityAsync(-0.5 if center_y < ver_part_size * 3 else 0.5, 0, 0, 1)
                                await asyncio.sleep(1)
                            # If it's in the 5th partition, it's centered vertically.

                            # If both horizontal and vertical are centered
                            if hor_part_size * 4 < center_x < hor_part_size * 6 and ver_part_size * 4 < center_y < ver_part_size * 6:
                                print("Target centered! Preparing to land...")
                                running = False
                                break

                # If no target detected, hover in place
                if not target_detected:
                    client.moveByVelocityAsync(3, 0, 0, 2)  # Move forward
                    await asyncio.sleep(2)  # Wait for 5 seconds

                # Display the processed image
                cv2.imshow("Drone Camera Feed", latest_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    running = False

            # Add a small delay to prevent tight looping
            await asyncio.sleep(0.1)

        logger.info("Main loop completed or target centered")

        # Lower altitude
        print("Lowering altitude...")
        current_position = client.getMultirotorState().kinematics_estimated.position
        client.moveToPositionAsync(current_position.x_val, current_position.y_val, -1, 2)
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
        logger.info("Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())