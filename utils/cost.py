
import abc

import torch


class CEnergy(abc.ABC):
    """
    Abstract base class for all supervised learning cost functions.

    Attributes:
        beta: Scalar weighting factor of the cost function
        target: Current target tensor for the cost function
    """
    def __init__(self, beta):
        super(CEnergy, self).__init__()
        self.beta = beta
        self.target = None

    @abc.abstractmethod
    def compute_energy(self, u_last):
        """
        Compute energy/loss given a prediction.

        Args:
            Activation of the last layer, i.e. the prediction
        """
        return

    def set_target(self, target):
        """
        Set new target tensor for the cost function.

        Args:
            target: target tensor
        """
        self.target = target


class CrossEntropy(CEnergy):
    """
    Cross entropy cost function.
    """
    def __init__(self, beta):
        super(CrossEntropy, self).__init__(beta)

    def compute_energy(self, u_last):
        """
        Compute cross-entropy loss given a prediction.

        Args:
            Activation of the last layer, i.e. the prediction
        """
        loss = torch.nn.functional.cross_entropy(u_last, self.target.double(), reduction='none')
        return self.beta * loss

    def set_target(self, target):
        if target is None:
            self.target = None
        else:
            # Need to transform target for the F.cross_entropy function
            self.target = target.argmax(dim=1)


class SquaredError(CEnergy):
    """
    Squared energy cost function.
    """
    def __init__(self, beta):
        super(SquaredError, self).__init__(beta)

    def compute_energy(self, u_last):
        """
        Compute mean squared error loss given a prediction.

        Args:
            Activation of the last layer, i.e. the prediction
        """
        
        loss = torch.nn.functional.mse_loss(u_last, self.target.double(), reduction='none') 
        return self.beta * torch.sum(loss, dim=1)
