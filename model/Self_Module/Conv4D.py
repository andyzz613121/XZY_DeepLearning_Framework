import torch
import torch.nn as nn
from torch.nn.modules.utils import _quadruple
import math
import torch.nn.functional as F

class Conv4d_seperate2X3D(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int = 3,
                 padding:int = 1,

                 bias=True):
        super(Conv4d_seperate2X3D, self).__init__()
        self.conv3D_1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        self.conv3D_2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            # nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w, H, W = x.size()
        # print(b, c, h, w, H, W)
        x1 = x.view(b, c, h*w, H, W)
        x2 = x.view(b, c, h, w, H*W).permute(0, 1, 4, 2, 3)
        print(x1.shape, x2.shape)
        # conv_3D_1 = self.conv3D_1(x1)
        conv_3D_2 = self.conv3D_2(x2).permute(0, 1, 3, 4, 2)
        # print(conv_3D_1.shape, conv_3D_2.shape, self.conv3D_2(x2).shape)
        # conv_3D_1 = conv_3D_1.view(b, -1, h, w, H, W)
        conv_3D_2 = conv_3D_2.view(b, -1, h, w, H, W)
        # conv_4D = conv_3D_1 * conv_3D_2
        # print(conv_4D.shape)
        return conv_3D_2

class Conv4d1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=True, drop_connect=0.):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) * 4 if isinstance(kernel_size, int) else kernel_size 
        self.padding = (padding,) * 4 if isinstance(padding, int) else padding 
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, *self.kernel_size))
        self.drop_connect = drop_connect
        if bias:
            self.bias = nn.Parameter(torch.rand(out_channels)) 
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        k1, k2, k3, k4 = self.kernel_size
        p1, p2, p3, p4 = self.padding
        input_pad = F.pad(input, (p4, p4, p3, p3, p2, p2, p1, p1))
        B, C_in, W2, H2, U2, V2 = input_pad.size()
        assert C_in == self.in_channels
        C_out = self.out_channels

        input_unfold = input_pad.as_strided(
            [B, C_in, k1, k2, k3, k4, W2-k1+1, H2-k2+1, U2-k3+1, V2-k4+1],
            [C_in*W2*H2*U2*V2, W2*H2*U2*V2, H2*U2*V2, U2*V2, V2, 1, H2*U2*V2, U2*V2, V2, 1]
        )
        weight = self.weight
        if self.drop_connect > 0:
            weight = F.dropout(weight, p=self.drop_connect, training=self.training)
        # print(weight.shape, input_unfold.shape)
        output = torch.einsum('oicdef,bicdefwhuv->bowhuv', (weight, input_unfold))
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1, 1)
        return output

    def extra_repr(self):
        s = '{}, {}, kernel_size={}, padding={}, bias={}'.format(
            self.in_channels, self.out_channels,
            self.kernel_size, self.padding, self.bias is not None,
        )
        if self.drop_connect > 0:
            s += ', drop_connect={}'.format(self.drop_connect)
        return s

class Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size = (3, 3, 3, 3),
                 stride = (1, 1, 1, 1),
                 padding = (0, 0, 0, 0),
                 dilation = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):
        super().__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::])
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight


    # def reset_parameters(self) -> None:
    #     nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.bias, -bound, bound)
    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out