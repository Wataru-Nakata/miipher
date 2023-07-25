from typing import Any, Optional
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from .model.miipher import Miipher
from omegaconf import DictConfig
from torch import nn
import hydra




class MiipherLightningModule(LightningModule):
    def __init__(self,cfg:DictConfig) -> None:
        super().__init__()

        self.miipher = Miipher(**cfg.model)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cfg = cfg
        self.save_hyperparameters()

    def training_step(self,batch,batch_idx) -> STEP_OUTPUT:
        phone_feature, speaker_feature, degraded_ssl_feature,clean_ssl_feature = batch["phone_feature"], batch['speaker_feature'], batch['degraded_ssl_feature'], batch['cleaned_ssl_feature']
        cleaned_feature, intermediates = self.miipher.forward(phone_feature,speaker_feature,degraded_ssl_feature)
        loss = self.criterion(intermediates,clean_ssl_feature)
        self.log('train/loss', loss,batch_size=phone_feature.size(0))
        return loss
    def validation_step(self,batch,batch_idx) -> STEP_OUTPUT | None:
        phone_feature, speaker_feature, degraded_ssl_feature,clean_ssl_feature = batch["phone_feature"], batch['speaker_feature'], batch['degraded_ssl_feature'], batch['cleaned_ssl_feature']
        cleaned_feature, intermediates = self.miipher.forward(phone_feature,speaker_feature,degraded_ssl_feature)
        loss = self.criterion(intermediates,clean_ssl_feature)
        return loss
    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.optimizers)
    def criterion(self,intermediates,target):
        loss =0
        for intermediate in intermediates:
            loss += self.mae_loss(intermediate,target)
            loss += self.mse_loss(intermediate,target)
            loss += self.mse_loss(intermediate,target) / (target.norm(p=2).pow(2))
        return loss
