import airsim
import cv2
import numpy as np
import threading
import time

def display_video_feed():
    while not mission_complete:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        cv2.imshow("Drone Camera", img_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

mission_complete = False

# Start video feed in a separate thread
video_thread = threading.Thread(target=display_video_feed)
video_thread.start()

# Takeoff
print("Taking off...")
client.moveByVelocityZAsync(0, 0, -0.5, 5).join()
time.sleep(5)

# Move to 15 meters height
print("Moving to target altitude...")
client.moveToZAsync(-15, 1).join()
print("Reached target altitude")

# Hover for a few seconds
time.sleep(5)

# Land
print("Landing...")
client.landAsync().join()
print("Landing complete")

# Disarm the drone
client.armDisarm(False)

# Signal that the mission is complete
mission_complete = True

# Wait for the video thread to finish
video_thread.join()

# Restore manual control
client.enableApiControl(False)