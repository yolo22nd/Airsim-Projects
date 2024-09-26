import torch

def load_model(model_path):
    try:
        model = torch.jit.load(model_path)
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None



def test_model(model, input_tensor):
    try:
        with torch.no_grad():  # Disable gradient computation
            output = model(input_tensor)
        print("Inference successful.")
        return output
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


if __name__ == "__main__":
    # model_path = "best_b.torchscript"
    model_path = "best_v5_x.torchscript"
    input_shape = (1, 3, 224, 224)  
    
    # Load the model
    model = load_model(model_path)
    
    if model is not None:
        # Create a dummy input tensor
        input_tensor = torch.randn(input_shape)
        
        # Test the model
        output = test_model(model, input_tensor)
        
        if output is not None:
            print("Model output:", output)