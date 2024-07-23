import airsim
import time

# Connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Take off
print("Taking off...")
client.takeoffAsync().join()

# Define the square path
side_length = 10  # meters
speed = 5  # meters per second

# Fly in a square pattern
print("Flying in a square pattern...")
client.moveByVelocityZAsync(speed, 0, -10, side_length / speed).join()
client.moveByVelocityZAsync(0, speed, -10, side_length / speed).join()
client.moveByVelocityZAsync(-speed, 0, -10, side_length / speed).join()
client.moveByVelocityZAsync(0, -speed, -10, side_length / speed).join()

# Hover
print("Hovering...")
client.hoverAsync().join()

# Land
print("Landing...")
client.landAsync().join()

# Disable API control and disarm
client.armDisarm(False)
client.enableApiControl(False)

print("Done!")
