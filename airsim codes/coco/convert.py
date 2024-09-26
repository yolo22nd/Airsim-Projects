import torch
from ultralytics import YOLO

# Load the YOLOv10 model
model = YOLO("yolov10x.pt")

# Set the model to evaluation mode
model.eval()

model.export()

# Trace the model
traced_model = torch.jit.script(model)

# Save the TorchScript model
traced_model.save("yolov10x_torchscript.pt")