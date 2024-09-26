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

def is_target_centered(center_x, center_y, x_threshold=12, y_threshold=12):

    return abs(center_x - img_size/2) < x_threshold and abs(center_y - img_size/2) < y_threshold

async def main():
    running = True
    max_duration =  120 # Maximum run time in seconds (increased to 5 minutes)
    start_time = time.time()

    try:
        # Takeoff
        print("Taking off...")
        client.takeoffAsync()
        await asyncio.sleep(3)  # Wait for takeoff completion

        pose = client.simGetVehiclePose()

        # Extract the position
        current_position = pose.position

        # Extract the coordinates (x, y, z)
        x = current_position.x_val
        y = current_position.y_val


        # Move to an initial altitude
        print("Moving to 7 meters altitude...")
        client.moveToPositionAsync(0, 0, -7, 5)
        await asyncio.sleep(5)  # Wait for 5 seconds

        count=0

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
                            center_y =  ((c1[1] + c2[1])- 50) / 2

                            # Check if target is centered
                            if is_target_centered(center_x, center_y):
                                if(count<3):
                                    count+=1
                                    print("centering...")
                                    continue
                                print("Target centered! Preparing to land...")
                                running = False
                                # pose = client.simGetVehiclePose()

                                # # Extract the position
                                # current_position = pose.position

                                # # Extract the coordinates (x, y, z)
                                # x = current_position.x_val
                                # y = current_position.y_val

                                break

                            # Initialize velocities
                            vel_x = 0
                            vel_y = 0

                            # Move drone based on target position (horizontal movement)
                            if center_x < 3 * img_size / 15:
                                print("left fast")
                                vel_y = -1  # Move lefter fast
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (0, img_size//2), (0, 0, 255), 2)
                            elif center_x < 5 * img_size / 15:
                                print("left slow")
                                vel_y = -0.7  # Move left slower
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//6, img_size//2), (0, 0, 255), 2)
                            elif center_x < 7 * img_size / 15:
                                print("left very slow")
                                vel_y = -0.3  # Move left very slow
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//4, img_size//2), (0, 0, 255), 2)
                            elif center_x > 12 * img_size / 15:
                                print("right fast")
                                vel_y = 1  # Move righter fast
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size, img_size//2), (0, 0, 255), 2)
                            elif center_x > 10 * img_size / 15:
                                print("right slow")
                                vel_y = 0.7  # Move right slower
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (5*img_size//6, img_size//2), (0, 0, 255), 2)
                            elif center_x > 8 * img_size / 15:
                                print("right very slow")
                                vel_y = 0.3  # Move right very slow
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (3*img_size//4, img_size//2), (0, 0, 255), 2)

                            # Vertical movement based on center_y
                            if center_y < 3 * img_size / 15:
                                print("up fast")
                                vel_x = 1  # Move up fast
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//2, 0), (0, 0, 255), 2)
                            elif center_y < 5 * img_size / 15:
                                print("up slow")
                                vel_x = 0.7  # Move up slower
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//2, img_size//6), (0, 0, 255), 2)
                            elif center_y < 7 * img_size / 15:
                                print("up very slow")
                                vel_x = 0.3  # Move up very slow
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//2, img_size//4), (0, 0, 255), 2)
                            elif center_y > 12 * img_size / 15:
                                print("down fast")
                                vel_x = -1  # Move down fast
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//2, img_size), (0, 0, 255), 2)
                            elif center_y > 10 * img_size / 15:
                                print("down slow")
                                vel_x = -0.7  # Move down slower
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//2, 5*img_size//6), (0, 0, 255), 2)
                            elif center_y > 8 * img_size / 15:
                                print("down very slow")
                                vel_x = -0.3  # Move down very slow
                                cv2.arrowedLine(latest_image, (img_size//2, img_size//2), (img_size//2, 3*img_size//4), (0, 0, 255), 2)

                            # Display the processed image
                            cv2.imshow("Drone Camera Feed", latest_image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                                running = False

                            # Execute the movement command with the final velocities
                            client.moveByVelocityAsync(vel_x, vel_y, 0, 2)
                            await asyncio.sleep(2)

                # If no target detected, hover in place
                if not target_detected:
                    print("surveying...")
                    client.moveByVelocityAsync(3, 0, 0, 2)  # Move forward
                    await asyncio.sleep(2)  # Wait for 5 seconds

            # Add a small delay to prevent tight looping
            await asyncio.sleep(0.1)

        logger.info("Main loop completed or target centered")
     
        # pose = client.simGetVehiclePose()

        # # Extract the position
        # current_position = pose.position

        # # Extract the coordinates (x, y, z)
        # x = current_position.x_val
        # y = current_position.y_val

        # Lower altitude
        print("Lowering altitude...")
        current_position = client.getMultirotorState().kinematics_estimated.position
        # client.moveToPositionAsync(current_position.x_val, current_position.y_val, -0.5, 2)
        # client.moveToPositionAsync(x, y, -0.5, 2)
        client.moveByVelocityAsync(0, 0, 3, 4)  # down
        await asyncio.sleep(3)

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