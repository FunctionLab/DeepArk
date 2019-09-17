"""
The DeepArk model and supporting classes.

This file was created on September 13, 2019
Author: Evan Cofer
"""
import numpy
import torch
import torch.nn


def _flip(x, dim):
    """
    Reverses the elements in a given dimension `dim` of the Tensor.
    source: https://github.com/pytorch/pytorch/issues/229
    """
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(
     x.size(0), x.size(1), -1)[:, getattr(
         torch.arange(x.size(1)-1, -1, -1),
         ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class Dropout1d(torch.nn.Module):
    """
    Spatial dropout layer used in DeepArk model.
    This layer randomly zeros whole channels of the input tensor.
    The channels to zero out are randomized on every forward call.

    Parameters
    ----------
    p: float
        The probability of a channel being set to zero during the forward
        pass of training.
    inplace: bool, optional
        Toggle whether this operation occurs in-place or not. Default
        is `False`.

    Attributes
    ----------
    p : float
        The probability of a channel being set to zero during the forward
        pass of training.
    inplace: bool
        Toggle whether this operation occurs in-place or not.

    """
    def __init__(self, p, inplace=False):
        super(Dropout1d, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        """
        Make predictions for some input sequence.

        Parameters
        ----------
        input: torch.Tensor
            The input tensor of encoded sequences. This should be a `torch.Tensor` of shape
            :math:`N \\times C \\times L`, where :math:`N` is the batch size, :math:`C` 
            is `input_conv_filters`, and :math:`L` is the length of the input sequence.

        Returns
        -------
        torch.Tensor
            Returns the input with some channels zeroed-out if training.

        """
        input = input.unsqueeze(-1) # Add W dim for 2d.
        input = torch.nn.functional.dropout2d(input, self.p, self.training, self.inplace) # Drop channels.
        input = input.squeeze(-1) # Remove unused W dim.
        return input

    def extra_repr(self):
        """
        String representation of the layer

        Returns
        -------
        str
            The string representation of the layer.

        """
        return "p={},inplace={}".format(self.p, self.inplace)


class ResidualBlock(torch.nn.Module):
    """
    This class implements the residual blocks used in the DeepArk model.

    Parameters
    ----------
    input_conv_filters : int
        The number of channels in the input tensors.
    output_conv_filters: int
        The number of channels in the output tensors.
    conv_kernel_size: int
        The length of the convolutional kernels to use.
    conv_kernel_dilation: int, optional
        The dilation to use in the convolutional kernels. The default is `1`.
    conv_layer_count: int, optional
        The number of convolutional layers to use in each residual block.
        The default is `2`.
    dropout_p: float, optional
        The dropout probability to use with the spatial dropout layers.
        The default is `0.` (i.e. no dropout).
    groups: int, optional
        The number of groups to use in the convolutional layers. The
        default is `1`.
    pool_kernel_size: int, optional
        The width of the maximum pooling kernel to use.  The default is `0`
        (i.e. no pooling).
    pool_kernel_dilation: int, optional
        The dilation of the maximum pooling kernel to use. Only utilized if
        `pool_kernel_size` is not 0. The default is `1`.


    Attributes
    ----------
    network: torch.nn.Module
        The convolutional block inside of the residual block.
    pool_network: torch.nn.Module
        The pooling layer used in the identity mapping of the residual network.
        If pooling is not used, this will be `torch.nn.Identity`.
    padding: int
        Half of the amount the sequence length is reduced by the convolutional
        layers.
    """
    def __init__(self,
                 input_conv_filters, 
                 output_conv_filters, 
                 conv_kernel_size,
                 conv_kernel_dilation=1,
                 conv_layer_count=2,
                 dropout_p=0.,
                 groups=1,
                 pool_kernel_size=0,
                 pool_kernel_dilation=1):
        super(ResidualBlock, self).__init__()
        self.network = list()
        self.padding = int((conv_kernel_size // 2) * conv_kernel_dilation) * conv_layer_count
        
        # Build pooling layers if necessary.
        if pool_kernel_size == 0:
            self.pool_network = torch.nn.Identity()
        else:
            self.pool_network = torch.nn.Sequential(torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size - (pool_kernel_size % 2)))
            self.network.append(torch.nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size - (pool_kernel_size % 2)))
            
        # Build conv layers.
        for i in range(conv_layer_count):
            if i != 0:
                input_conv_filters = output_conv_filters
            self.network.extend([torch.nn.BatchNorm1d(input_conv_filters),
                                 torch.nn.ReLU(),
                                 Dropout1d(p=dropout_p),
                                 torch.nn.Conv1d(input_conv_filters, 
                                           output_conv_filters, 
                                           kernel_size=conv_kernel_size,
                                           dilation=conv_kernel_dilation,
                                           padding=0,
                                           groups=groups)])
            
        self.network = torch.nn.Sequential(*self.network)
       
    def forward(self, input):
        """
        Make predictions for some input sequence.

        Parameters
        ----------
        input: torch.Tensor
            The input tensor of encoded sequences. This should be a `torch.Tensor` of shape
            :math:`N \\times C \\times L`, where :math:`N` is the batch size, :math:`C` 
            is `input_conv_filters`, and :math:`L` is the length of the input sequence.

        Returns
        -------
        torch.Tensor
            Returns the predictions of the residual block for the given inputs.

        """
        identity_input = self.pool_network.forward(input)
        return self.network.forward(input) + identity_input[:, :, self.padding:-self.padding]
    
    def extra_repr(self):
        """
        String representation of the residual block..

        Returns
        -------
        str
            The string representation of the residual block.

        """
        return "network={}".format(self.network.extra_repr())
    

class DeepArkModel(torch.nn.Module):
    """
    This class implements the DeepArk model.

    Parameters
    ----------
    sequence_length: int
        The length of the sequences that the network will make predictions for. This
        should almost always be `5797`.
    n_features: int
        The number of features that the network will make predictions for.
    dropout: float
        The dropout probability to use in the spatial dropout layers.

    Attributes
    ----------
    n_features: int
        The number of output features.
    sequence_length: int
        The length of the input sequence.
    residual_networks: torch.nn.ModuleList
        The set of residual blocks that form the convolutional network.
    trailer_network: torch.nn.Sequential
        Classification network applied to the output of the convolutional networks.
        
    """
    def __init__(self, sequence_length, n_features, dropout):
        super(DeepArkModel, self).__init__()
        self.n_features = n_features
        self.sequence_length = sequence_length

        # Dropout params.
        block_count = 5
        channels = [4, 160, 320, 480, 560, 720]
        self.residual_networks = list()
        for i in range(block_count):
            pool_kernel_size = 5 if i > 0 else 0
            cur_layers = list()
            if channels[i] != channels[i + 1]:
                cur_layers.extend([
                    torch.nn.BatchNorm1d(channels[i]),
                    torch.nn.ReLU(),
                    torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=1)])
            cur_layers.append(ResidualBlock(channels[i + 1], 
                                            channels[i + 1],
                                            conv_kernel_size=9, 
                                            conv_kernel_dilation=1,
                                            conv_layer_count=2,
                                            dropout_p=dropout,
                                            groups=1,
                                            pool_kernel_size=pool_kernel_size))
            self.residual_networks.append(torch.nn.Sequential(*cur_layers))
        self.residual_networks = torch.nn.ModuleList(self.residual_networks)

        self.trailer_network = torch.nn.Sequential(
            torch.nn.Conv1d(channels[-1],
                            n_features,
                            kernel_size=1,
                            dilation=1,
                            padding=0),
            torch.nn.Sigmoid())

    def _forward(self, input):
        """
        This method should be called on the forward sequence and its reverse complement. It is then

        Parameters
        ----------
        input: torch.Tensor
            The input tensor of genomic sequences. This should be a `torch.Tensor` of shape
            `N \\times 4 \\times 5797`.

        Returns
        -------
        torch.Tensor
            Returns the model prediction for the input tensor.

        """
        bs = input.shape[0]
        cur = input
        for i in range(len(self.residual_networks)):
            cur = self.residual_networks[i].forward(cur)
        ret = self.trailer_network.forward(cur)
        return ret.reshape(bs, -1)

    def forward(self, input):
        """
        Makes predictions for some labeled input data.

        Parameters
        ----------
        input: torch.Tensor
            The input tensor of genomic sequences. This should be a `torch.Tensor` of shape
            `N \\times 4 \\times 5797`.

        Returns
        -------
        torch.Tensor
            Returns the average of model prediction on the sequence and its reverse complement.

        """
        bs = input.shape[0]
        reverse_input = _flip(_flip(input, 1), 2)
        output = self._forward(input)
        output_from_rev = self._forward(reverse_input)
        output_from_rev = _flip(output_from_rev.reshape(bs, self.n_features, -1), 2).reshape(bs, -1)
        return (output + output_from_rev) / 2


def criterion():
    """
    Get the loss function used to train DeepArk.


    Returns
    -------
    torch.nn.BCELoss
        The objective function used to train DeepArk.
    """
    return torch.nn.BCELoss()


def get_optimizer(lr):
    """
    Get the method used to optimize and train the DeepArk models.

    Note that the parameters in this function (e.g. weight decay) may
    differ depending on the specific organism that the DeepArk model 
    was trained on. Refer to the manuscript for definitive values used
    for training each model.

    Parameters
    ----------
    lr: float
        The learning rate at the start of training.

    Returns
    -------
    torch.optim.SGD
        The optimizer used to train DeepArk.
    """
    return (torch.optim.SGD,
        {"lr": lr, "weight_decay": 1e-6, "momentum": 0.9})
