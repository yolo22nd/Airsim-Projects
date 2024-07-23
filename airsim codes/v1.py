import airsim
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import asyncio

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Global variables
mission_active = False
video_active = False

def capture_image():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    response = responses[0]
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3)
    return img_rgb

async def start_mission():
    global mission_active
    mission_active = True

    # Take off
    await client.takeoffAsync().join()

    # Define the star-shaped path
    waypoints = [
        [0, 30, -10], [14.4, 9.6, -10], [30, 30, -10], [19.2, 0, -10],
        [30, -30, -10], [14.4, -9.6, -10], [0, -30, -10], [-14.4, -9.6, -10],
        [-30, -30, -10], [-19.2, 0, -10], [-30, 30, -10], [-14.4, 9.6, -10]
    ]

    for point in waypoints:
        if not mission_active:
            break
        await client.moveToPositionAsync(point[0], point[1], point[2], 5).join()

    # Land
    await client.landAsync().join()

async def stop_mission():
    global mission_active
    mission_active = False
    await client.hoverAsync().join()

async def return_home():
    await client.goHomeAsync().join()

def telemetry_data():
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    velocity = state.kinematics_estimated.linear_velocity
    altitude = -position.z_val

    telemetry_text.set(f"Position: x={position.x_val:.2f}, y={position.y_val:.2f}, z={altitude:.2f}\n"
                       f"Velocity: x={velocity.x_val:.2f}, y={velocity.y_val:.2f}, z={velocity.z_val:.2f}")

    root.after(1000, telemetry_data)

def video_feed():
    global video_active
    while video_active:
        frame = capture_image()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (320, 240))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.imencode('.png', img)[1].tobytes()
        photo = tk.PhotoImage(data=img)
        video_label.config(image=photo)
        video_label.image = photo
        time.sleep(0.1)

def start_video_feed():
    global video_active
    video_active = True
    thread = Thread(target=video_feed)
    thread.start()

def stop_video_feed():
    global video_active
    video_active = False

def async_task_wrapper(task):
    asyncio.create_task(task())

# GUI setup
root = tk.Tk()
root.title("Drone Control")

telemetry_text = tk.StringVar()
telemetry_label = tk.Label(root, textvariable=telemetry_text, justify='left')
telemetry_label.pack()

start_button = tk.Button(root, text="Start Mission", command=lambda: async_task_wrapper(start_mission))
start_button.pack()

stop_button = tk.Button(root, text="Stop Mission", command=lambda: async_task_wrapper(stop_mission))
stop_button.pack()

return_button = tk.Button(root, text="Return to Home", command=lambda: async_task_wrapper(return_home))
return_button.pack()

start_video_button = tk.Button(root, text="Start Video Feed", command=start_video_feed)
start_video_button.pack()

stop_video_button = tk.Button(root, text="Stop Video Feed", command=stop_video_feed)
stop_video_button.pack()

video_label = tk.Label(root)
video_label.pack()

telemetry_data()  # Start updating telemetry data
root.mainloop()

client.armDisarm(False)
client.enableApiControl(False)
