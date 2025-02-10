from abc import ABCMeta, abstractmethod
import torch

# custom library imports
from pipelinegen.core.types import TorchModelMixin


class EncoderDecoderMixin(TorchModelMixin):
    @abstractmethod
    def encode(self, data: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, data: torch.Tensor) -> torch.Tensor: ...
