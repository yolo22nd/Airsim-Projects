# simple takeoff and land

import airsim
import cv2
import numpy as np
import threading
import time
import asyncio

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Global variables
running = True
latest_image = None
image_lock = threading.Lock()

def get_image():
    responses = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def update_image():
    global latest_image, running
    while running:
        new_image = get_image()
        with image_lock:
            latest_image = new_image
        time.sleep(0.1)  # ~10 FPS

def display_image():
    global latest_image, running
    while running:
        with image_lock:
            if latest_image is not None:
                cv2.imshow("Drone Camera", latest_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
    cv2.destroyAllWindows()

# Start image update thread
update_thread = threading.Thread(target=update_image)
update_thread.start()

# Start display thread
display_thread = threading.Thread(target=display_image)
display_thread.start()

async def main():
    try:
        # Takeoff
        print("Taking off...")
        client.takeoffAsync()
        await asyncio.sleep(2)  # Allow time for takeoff

        # Move to 15 meters height
        print("Moving to 15 meters altitude...")
        client.moveToPositionAsync(0, 0, -15, 5)
        await asyncio.sleep(5)  # Allow time to reach the target altitude

        # Hover for 5 seconds
        print("Hovering...")
        await asyncio.sleep(5)

        
        # Move to 1 meters height
        print("Moving to 1 meters altitude...")
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
        print("Operation interrupted by user")

    finally:
        # Cleanup
        running = False
        update_thread.join()
        display_thread.join()
        cv2.destroyAllWindows()
        client.armDisarm(False)
        client.enableApiControl(False)
        print("Mission complete")

# Run the main function
asyncio.run(main())
