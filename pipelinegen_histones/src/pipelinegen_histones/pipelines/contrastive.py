from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from typing import Dict, Any, Tuple

from pipelinegen.core.pipelines.factory import AbstractPipelineBuilder
from pipelinegen.core.pipelines.prototype import (
    LightningTrainerPrototype,
    PipelineStage,
)
from pipelinegen.core.pipelines.objectives.simple_objective import (
    SimpleObjectivePipelineStage,
)
from pipelinegen_histones.models.autoencoders import EncoderDecoderMixin


class ContrastiveLoss(nn.Module):
    """Contrastive Loss Function"""

    def __init__(self, temperature: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def _random_masking(self, y_hat):
        """Randomly mask some elements in the input tensor"""
        mask = (torch.rand(y_hat.size()) < 0.5).to(y_hat.device)
        return y_hat * mask

    def _generate_positive_pairs(self, data):
        """Generate positive pairs from the predictions"""
        masked = self._random_masking(data)
        positive_pairs = torch.cat([masked, data], dim=0)
        sim = torch.mm(positive_pairs, positive_pairs.t())
        labels = torch.arange(positive_pairs.size(0)).to(positive_pairs.device)
        return sim, labels

    def forward(self, data):
        """Compute the contrastive loss"""
        sim, labels = self._generate_positive_pairs(data)
        sim = sim / self.temperature
        loss = self.criterion(sim, labels)
        return loss


class AutoencoderContrastiveObjective(LightningTrainerPrototype):
    """Contrastive Objective for training"""

    def __init__(
        self, model: EncoderDecoderMixin, metrics: nn.ModuleList, config: Dict[str, Any]
    ):
        self._model: EncoderDecoderMixin
        super().__init__(model, metrics, config)
        criterion = config.get("criterion", "MSELoss")
        if hasattr(nn, criterion):
            self._criterion = getattr(nn, criterion)()
        else:
            raise ValueError(f"Criterion {criterion} not found in nn module.")
        self._contrastive_loss = ContrastiveLoss(
            temperature=config.get("temperature", 0.1)
        )
        self.w1 = config.get("weight", 1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """
        Defines the training step for the model. Uses the contrastive loss function
        and the reconstruction loss to train the model.

        Args:
            batch: data batch
            batch_idx: batch index
        Returns:
            STEP_OUTPUT: output of the model

        """
        data, _ = batch
        _ = batch_idx

        latent_space = self._model.encode(data)
        reconstructions = self._model.decode(latent_space)
        recon_loss = self._criterion(reconstructions, data)
        contrastive_loss = self._contrastive_loss(latent_space)
        loss = self.w1 * recon_loss + (1 - self.w1) * contrastive_loss

        return {
            "loss": loss,
            "y_hat": reconstructions,
            "y": data,
        }

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        """Defines a shared step for validation and testing.
        Skips the contrastive loss and only computes the reconstruction loss.

        Args:
            batch: data batch
            batch_idx: batch index

        Returns:
            STEP_OUTPUT: output of the model

        """
        data, _ = batch
        reconstructions = self._model(data)
        loss = self._criterion(reconstructions, data)

        return {
            "loss": loss,
            "y_hat": reconstructions,
            "y": data,
        }


class ContrastivePipelineBuilder(AbstractPipelineBuilder):
    """Pipeline Builder for Contrastive Autoencoder"""

    name = "contrastive_autoencoder"

    def build(self, dataset_config, objective_config) -> PipelineStage:
        return SimpleObjectivePipelineStage(
            dataset_config=dataset_config,
            objective_config=objective_config,
            objective=AutoencoderContrastiveObjective,
        )
