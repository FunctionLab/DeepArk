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


class DeepArkModel(torch.nn.Module):
    """
    This class implements the DeepArk model.

    Parameters
    ----------
    sequence_length: int
        The length of the sequences that the network will make predictions for. This
        should almost always be `4095`.
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
    network: torch.nn.Sequential
        The convolutional network.
        
    """
    def __init__(self, sequence_length, n_features, dropout):
        super(DeepArkModel, self).__init__()
        self.n_features = n_features
        self.sequence_length = sequence_length

        # Dropout params.
        block_count = 5
        channels = [160, 320, 480, 560, 720]
        self.network = torch.nn.Sequential(
            ## Block 1.
            torch.nn.BatchNorm1d(4),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(4, channels[0], kernel_size=3),
            #
            torch.nn.BatchNorm1d(channels[0]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[0], channels[0], kernel_size=5),
            #
            torch.nn.BatchNorm1d(channels[0]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[0], channels[0], kernel_size=5),
            torch.nn.MaxPool1d(kernel_size=5, stride=4),
            
            ## Block 2.
            torch.nn.BatchNorm1d(channels[0]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[0], channels[1], kernel_size=5),
            #
            torch.nn.BatchNorm1d(channels[1]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[1], channels[1], kernel_size=5),
            torch.nn.MaxPool1d(kernel_size=5, stride=4),
            
            ## Block 3.
            torch.nn.BatchNorm1d(channels[1]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[1], channels[2], kernel_size=5),
            #
            torch.nn.BatchNorm1d(channels[2]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[2], channels[2], kernel_size=5),
            torch.nn.MaxPool1d(kernel_size=5, stride=4),
            
            ## Block 4.
            torch.nn.BatchNorm1d(channels[2]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[2], channels[3], kernel_size=5),
            #
            torch.nn.BatchNorm1d(channels[3]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[3], channels[3], kernel_size=5),
            torch.nn.MaxPool1d(kernel_size=5, stride=4),
            
            ## Block 5.
            torch.nn.BatchNorm1d(channels[3]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[3], channels[4], kernel_size=5),
            #
            torch.nn.BatchNorm1d(channels[4]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[4], channels[4], kernel_size=5),
            #
            torch.nn.BatchNorm1d(channels[4]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[4], channels[4], kernel_size=5),
            
            ## Classifier.
            torch.nn.BatchNorm1d(channels[4]),
            torch.nn.ReLU(inplace=True),
            Dropout1d(p=dropout),
            torch.nn.Conv1d(channels[4], n_features, kernel_size=1, dilation=1, padding=0),
            torch.nn.Sigmoid())


    def _forward(self, input):
        """
        This method should be called on the forward sequence and its reverse complement. It is then

        Parameters
        ----------
        input: torch.Tensor
            The input tensor of genomic sequences. This should be a `torch.Tensor` of shape
            `N \\times 4 \\times 4095`.

        Returns
        -------
        torch.Tensor
            Returns the model prediction for the input tensor.

        """
        bs = input.shape[0]
        ret = self.network.forward(input)
        return ret.reshape(bs, -1)

    def forward(self, input):
        """
        Makes predictions for some labeled input data.

        Parameters
        ----------
        input: torch.Tensor
            The input tensor of genomic sequences. This should be a `torch.Tensor` of shape
            `N \\times 4 \\times 4095`.

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
