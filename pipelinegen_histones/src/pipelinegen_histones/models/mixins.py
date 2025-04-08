from abc import ABCMeta, abstractmethod
import torch
from torch import nn


class EncoderDecoderMixin(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, data: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, data: torch.Tensor) -> torch.Tensor: ...
