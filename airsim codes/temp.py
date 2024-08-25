import torch

# Load your TorchScript model
model = torch.jit.load("best.torchscript")

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor
# Assuming the model expects an input tensor with 3 channels (RGB)
dummy_input = torch.randn(1, 3, 640, 640)  # (batch_size, channels, height, width)

# Move the dummy input to the same device as the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dummy_input = dummy_input.to(device)
model.to(device)

# Pass the dummy input through the model
with torch.no_grad():
    output = model(dummy_input)

# Check if the output is a tuple and print the shapes of its elements
if isinstance(output, tuple):
    for i, out in enumerate(output):
        print(f"Output element {i} shape:", out.shape)
