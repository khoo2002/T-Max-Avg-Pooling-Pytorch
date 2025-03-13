# tmaxavg_pool.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TMaxAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, T=0.9, learnable_T=True):
        """
        T-Max-Avg Pooling Layer.
        
        Args:
            kernel_size (int or tuple): Size of the pooling window.
            stride (int or tuple, optional): Stride of the pooling window. Defaults to kernel_size.
            padding (int or tuple, optional): Padding on each side.
            T (float): The threshold multiplier. It multiplies the maximum value in each patch to 
                       set a threshold. Only values above this threshold are averaged.
            learnable_T (bool): If True, T becomes a learnable parameter.
        """
        super(TMaxAvgPool2d, self).__init__()
        # Ensure kernel_size is a tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        # Ensure stride is a tuple; if not provided, default to kernel_size.
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride

        # Ensure padding is a tuple
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        
        if learnable_T:
            self.T = nn.Parameter(torch.tensor(T, dtype=torch.float32))
        else:
            self.register_buffer('T', torch.tensor(T, dtype=torch.float32))

    def forward(self, x):
        B, C, H, W = x.size()
        # Extract patches from the input tensor
        patches = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        k = self.kernel_size[0] * self.kernel_size[1]
        patches = patches.view(B, C, k, -1)  # (B, C, patch_size, L)

        # Get maximum value per patch
        max_val, _ = patches.max(dim=2, keepdim=True)
        # Compute threshold value: T * (max value)
        threshold_value = self.T * max_val

        # Mask: only values above threshold will be considered
        mask = patches >= threshold_value

        # Sum and count the selected elements
        selected_sum = (patches * mask.float()).sum(dim=2)
        count = mask.float().sum(dim=2)
        
        # Average selected elements; fallback to max if none selected
        avg_val = selected_sum / (count + 1e-6)
        avg_val = torch.where(count > 0, avg_val, max_val.squeeze(2))
        
        # Compute the output spatial dimensions
        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        out = avg_val.view(B, C, H_out, W_out)
        return out
