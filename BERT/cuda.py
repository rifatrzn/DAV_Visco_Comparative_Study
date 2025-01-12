import torch

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    # Print the name of the GPU being used
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    print('CUDA is not available. Using CPU.')