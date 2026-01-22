import torch
import torch.nn as nn

"""
Simple Inference – PyTorch Implementation

Description:
This function runs inference on a trained PyTorch linear model (nn.Linear).
Given an input tensor and a model already loaded on the GPU, it computes
the forward pass and stores the result in the provided output tensor.

The linear model performs the operation:

    output = input @ weight.T + bias

where:
- weight has shape [output_size, input_size]
- bias has shape [output_size]

Constraints:
- Uses PyTorch’s built-in operations only.
- The solve function signature remains unchanged.
- The final result is written into the output tensor.
- The model is already on the GPU and ready for inference.
"""

# input, model, and output are on the GPU
def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    # Disable gradient tracking since this is inference only
    with torch.no_grad():
        # Run the forward pass
        result = model(input)

        # Store the result in the provided output tensor
        output.copy_(result)
