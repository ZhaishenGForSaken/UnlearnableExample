import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDNN version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("CUDA is available. Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
