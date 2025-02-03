import torch
import torch.nn.functional as F

from torch.nn import Conv2d

img = torch.rand(6, 6)
print("Image = \n", img)

#pytorch expects a 4d tensor as the input
img = img.unsqueeze(0).unsqueeze(0)
print(img.shape)

kernel = torch.ones(3, 3).unsqueeze(0).unsqueeze(0)
print("Kernel =\n", kernel)

output = F.conv2d(img, kernel, stride=1, padding=0)
print("Output after Convolution -\n", output)
print("\nShape of output = ", output.shape)

out_stride2 = F.conv2d(img, kernel, stride=2, padding=0)
print("Shape of op with stride 2 = ", out_stride2.shape)

# Question 2
nn_conv = Conv2d(1, 3, 3, 1, bias=0)
out2 = nn_conv(img)
print("\n\nShape of output with nn Conv2d = ", out2.shape)

# achieving 3 channel output with F.conv2d
kernel = torch.ones(3, 1, 3, 3)  # Shape: [out_channels, in_channels, kernel_height, kernel_width]
print("\nNew kernel shape for 3 channel op = \n", kernel.shape)

out3 = F.conv2d(img, kernel, stride=1, padding=0)
print("\nShape of output = ", out3.shape)