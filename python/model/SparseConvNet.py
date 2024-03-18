import torch.nn as nn
import torch

class SparseConvNet(nn. Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize the layer with given params

        Args:
            in_channels: # channels that the input has.
            out_channels: # channels that the output will have.
            kernel_size: height and width of the kernel in pixels.
            stride: # pixels between adjacent receptive fields in both
                horizontal and vertical.
            padding: # pixels that is used to zero-pad the input.
        """
        super(SparseConvNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = torch.Tensor(out_channels,
                                   in_channels,
                                   kernel_size,
                                   kernel_size)
        self.bias = torch.Tensor(out_channels)
        
        # Initialize parameters
        self.init_params()
        
    
    def init_params(self,std=0.7071):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias will be zeros.
        
        Args:
            std: Standard deviation of Gaussian distribution (default: 0.7071)
        """

        self.weight = std * torch.randn_like(self.weight)
        self.bias = torch.rand_like(self.bias)

    def forward(self, x, mask):
        """
        Forward pass of convolutional layer
        
        Args:
            x: input tensor which has a shape of (N, C, H, W)
            mask: Indicates which pixels in the input are invalid or not has a shape of (N, H, W) 

        Returns:
            given these definitions:
            H' = 1 + (H + 2 * padding - kernel_size) / stride
            W' = 1 + (W + 2 * padding - kernel_size) / stride
            
            output:
            y: output tensor which has a shape of (N, F, H', W')  
            pooledMask: updated mask of size (N, H', W')
        """
        

        # Pad the input
        x_padded = torch.nn.functional.pad(x, [self.padding] * 4)
        mask_padded = torch.nn.functional.pad(mask, [self.padding] * 4)
        
        # Cache input to use in backward pass
        self.cache_mask = mask_padded
        self.cache_input = x_padded

        # Unpack the needed dimensions
        N, _, H, W = x.shape

        # Calculate output height and width
        Hp = 1 + (H + 2 * self.padding - self.kernel_size) // self.stride
        Wp = 1 + (W + 2 * self.padding - self.kernel_size) // self.stride

        # Create an empty output to fill in
        y = torch.empty((N, self.out_channels, Hp, Wp), dtype=x.dtype, device=x.device)
        pooledMask = torch.empty((N, Hp, Wp), dtype=x.dtype, device=x.device)

        # Loop through each of the output value
        # One can find looping on input more easy but I find it more convenient
        # to loop over the output and find the corresponding input patch.
        for i in range(Hp):
            for j in range(Wp):
                # Calculate offsets on the input
                h_offset = i * self.stride
                w_offset = j * self.stride

                # Get the corresponding window of the input
                convWindow = x_padded[:, :, h_offset:h_offset+self.kernel_size, w_offset:w_offset+self.kernel_size] 
                
                # mask pooling 
                poolingWindow = mask_padded[:, h_offset:h_offset+self.kernel_size, w_offset:w_offset+self.kernel_size]
                pooledMask[:, i, j], _ = poolingWindow.max(dim=1)
                
                
                # Loop through each input sample to calculate convolution of
                # each filter and and current window of the input normalize this using the mask
                valid_pixels_amount = poolingWindow.sum()
                if valid_pixels_amount > 0:
                    for k in range(N):
                        y[k, :, i, j] =((poolingWindow[k] * convWindow[k] * self.weight).sum(dim=(1, 2, 3)) / valid_pixels_amount) + self.bias


        return y, pooledMask
    
    def backward(self, dupstream):
        """
        Backward pass of convolutional layer: calculate gradients of loss with
        respect to weight and bias and return downstream gradient dx.
        
        Args:
            dupstream: Gradient of loss with respect to output of this layer.

        Returns:
            dx: Gradient of loss with respect to input of this layer.
        """

        # You don't need to implement the backward pass. Instead we give it to
        # you the solution.

        # Unpack cache
        mask_padded = self.cache_mask # This is for the mask
        x_padded = self.cache_input #This is for the convLayer

        # Create an empty dx tensor to accumulate the gradients on. Keep in mind
        # that it has a size according to padded input
        dx_padded = torch.zeros_like(x_padded)

        # Also initialize the weight gradients as zeros
        self.weight_grad = torch.zeros_like(self.weight)

        # Unpack needed dimensions
        N, _, Hp, Wp = dupstream.shape

        # Loop through dupstream
        for i in range(Hp):
            for j in range(Wp):

                # Calculate offset for current window on input
                h_offset = i * self.stride
                w_offset = j * self.stride

                # Get current window of input and gradient of the input
                window = x_padded[:, :, h_offset:h_offset+self.kernel_size, w_offset:w_offset+self.kernel_size]
                dwindow = dx_padded[:, :, h_offset:h_offset+self.kernel_size, w_offset:w_offset+self.kernel_size]

                mask_window = mask_padded[:, h_offset:h_offset+self.kernel_size, w_offset:w_offset+self.kernel_size]
                # Walk through each sample of the input and accumulate gradients
                # of both input and weight
                valid_pixels_amount = mask_window.sum()
                if valid_pixels_amount > 0:
                    for k in range(N):
                        dwindow[k] += (mask_window * self.weight * dupstream[k, :, i, j].view(-1, 1, 1, 1)).sum(dim=0) / valid_pixels_amount
                        self.weight_grad += window[k].view(1, self.in_channels, self.kernel_size, self.kernel_size) * dupstream[k, :, i, j].view(-1, 1, 1, 1)
        # Calculate actual size of input height and width
        H = x_padded.shape[2] - 2 * self.padding
        W = x_padded.shape[3] - 2 * self.padding

        # Unpad dx
        dx = dx_padded[:, :, self.padding:self.padding+H, self.padding:self.padding+W]

        # Calculate bias gradients
        self.bias_grad = dupstream.sum(dim=(0, 2, 3))

        return dx