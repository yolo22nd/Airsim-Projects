import airsim
import cv2
import numpy as np
import time
from threading import Thread

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
client.takeoffAsync().join()

# Global variable to keep the video feed running
video_active = True

def capture_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    return img_rgb

def video_feed():
    while video_active:
        frame = capture_image()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Drone Camera Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def perform_square_maneuver():
    altitude = -15  # 15 meters above the ground
    waypoints = [
        [0, 0, altitude], [20, 0, altitude], [20, 20, altitude], [0, 20, altitude], [0, 0, altitude]
    ]

    for point in waypoints:
        try:
            client.moveToPositionAsync(point[0], point[1], point[2], 5).join()
        except Exception as e:
            print(f"Error moving to position {point}: {e}")
        time.sleep(1)

    # Return to home and land
    client.goHomeAsync().join()
    client.landAsync().join()

def main():
    # Start the video feed in a separate thread
    video_thread = Thread(target=video_feed)
    video_thread.start()

    # Perform the square maneuver
    perform_square_maneuver()

    # Cleanup
    global video_active
    video_active = False
    video_thread.join()
    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()
