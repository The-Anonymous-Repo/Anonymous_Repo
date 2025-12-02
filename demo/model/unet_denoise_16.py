import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet16(nn.Module):
    """
    UNet See in Dark for 4×4 CFA patterns
    
    Supports 16-channel input/output for 4×4 Color Filter Array patterns like:
    - RGBW (Red-Green-Blue-White)
    - RCCB (Red-Cyan-Cyan-Blue) 
    - Multi-spectral CFA
    - Custom 4×4 patterns
    
    Args:
        in_nc (int): Input channels, default 16 for 4×4 CFA
        out_nc (int): Output channels, default 16 for 4×4 CFA
        nf (int): Base number of filters, default 32
    """
    
    def __init__(self, in_nc=16, out_nc=16, nf=32):
        super(UNet16, self).__init__()
        
        self.conv1_1 = nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3_1 = nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4_1 = nn.Conv2d(nf * 4, nf * 8, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5_1 = nn.Conv2d(nf * 8, nf * 16, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(nf * 16, nf * 16, kernel_size=3, stride=1, padding=1)

        self.upv6 = nn.ConvTranspose2d(nf * 16, nf * 8, 2, stride=2)
        self.conv6_1 = nn.Conv2d(nf * 16, nf * 8, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1)

        self.upv7 = nn.ConvTranspose2d(nf * 8, nf * 4, 2, stride=2)
        self.conv7_1 = nn.Conv2d(nf * 8, nf * 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1)

        self.upv8 = nn.ConvTranspose2d(nf * 4, nf * 2, 2, stride=2)
        self.conv8_1 = nn.Conv2d(nf * 4, nf * 2, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1)

        self.upv9 = nn.ConvTranspose2d(nf * 2, nf, 2, stride=2)
        self.conv9_1 = nn.Conv2d(nf * 2, nf, kernel_size=3, stride=1, padding=1)
        self.conv9_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)

        self.conv10_1 = nn.Conv2d(nf, out_nc, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def check_img_size(self, x, ds_scale=16):
        """
        Ensure input size is divisible by downsampling scale × 4 (for 4×4 CFA)
        Modified for 4×4 CFA patterns which need 64-pixel alignment
        """
        effective_scale = ds_scale * 4  # 64 for 4×4 CFA
        mod_pad_h = effective_scale - x.shape[2] % effective_scale
        mod_pad_w = effective_scale - x.shape[3] % effective_scale
        
        # Only pad if necessary
        if mod_pad_h == effective_scale:
            mod_pad_h = 0
        if mod_pad_w == effective_scale:
            mod_pad_w = 0
            
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input 4×4 CFA packed tensor [B, 16, H/4, W/4]
            
        Returns:
            torch.Tensor: Denoised output [B, 16, H/4, W/4]
        """
        # Store original dimensions
        h, w = x.shape[2:]
        
        # Ensure proper size alignment
        x = self.check_img_size(x)

        # Encoder path
        conv1 = self.relu(self.conv1_1(x))
        conv1 = self.relu(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2_1(pool1))
        conv2 = self.relu(self.conv2_2(conv2))
        pool2 = self.pool1(conv2)  # Reuse pool1 (same as original)

        conv3 = self.relu(self.conv3_1(pool2))
        conv3 = self.relu(self.conv3_2(conv3))
        pool3 = self.pool1(conv3)  # Reuse pool1 (same as original)

        conv4 = self.relu(self.conv4_1(pool3))
        conv4 = self.relu(self.conv4_2(conv4))
        pool4 = self.pool1(conv4)  # Reuse pool1 (same as original)

        # Bottleneck
        conv5 = self.relu(self.conv5_1(pool4))
        conv5 = self.relu(self.conv5_2(conv5))

        # Decoder path with skip connections
        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)  # Concatenate skip connection
        conv6 = self.relu(self.conv6_1(up6))
        conv6 = self.relu(self.conv6_2(conv6))

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)  # Concatenate skip connection
        conv7 = self.relu(self.conv7_1(up7))
        conv7 = self.relu(self.conv7_2(conv7))

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)  # Concatenate skip connection
        conv8 = self.relu(self.conv8_1(up8))
        conv8 = self.relu(self.conv8_2(conv8))

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)  # Concatenate skip connection
        conv9 = self.relu(self.conv9_1(up9))
        conv9 = self.relu(self.conv9_2(conv9))

        # Output layer
        out = self.conv10_1(conv9)
        
        # Crop back to original size
        return out[:, :, :h, :w]

    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': 'UNetSeeInDark4x4',
            'input_channels': 16,
            'output_channels': 16,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        return info


if __name__ == "__main__":
    """
    Test the UNetSeeInDark4x4 model
    """
    kwargs = {
        'in_nc': 16,
        'out_nc': 16,
        'nf': 32
    }

    model = UNet16(**kwargs)
    rand_input = torch.randn(1, 16, 256, 256)
    rand_output = model(rand_input)
    print(f"Random input shape: {rand_input.shape}")
    print(f"Random output shape: {rand_output.shape}")