from typing import Any, Optional
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from .model.miipher import Miipher
from omegaconf import DictConfig
from torch import nn
import torch
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
        phone_feature, speaker_feature, degraded_ssl_feature,clean_ssl_feature = batch["phone"], batch['speaker'], batch['degraded_ssl'], batch['clean_ssl']
        ssl_feature_lengths = batch['ssl_lengths']
        cleaned_feature, intermediates = self.miipher.forward(phone_feature,speaker_feature,degraded_ssl_feature,ssl_feature_lengths)
        loss = self.criterion(intermediates,clean_ssl_feature)
        self.log('train/loss', loss,batch_size=phone_feature.size(0))
        return loss
    def validation_step(self,batch,batch_idx) -> STEP_OUTPUT | None:
        phone_feature, speaker_feature, degraded_ssl_feature,clean_ssl_feature = batch["phone"], batch['speaker'], batch['degraded_ssl'], batch['clean_ssl']
        ssl_feature_lengths = batch['ssl_lengths']
        cleaned_feature, intermediates = self.miipher.forward(phone_feature,speaker_feature,degraded_ssl_feature,ssl_feature_lengths)
        loss = self.criterion(intermediates,clean_ssl_feature)
        self.log('val/loss', loss,batch_size=phone_feature.size(0))
        return loss
    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.optimizers,params=self.parameters())
    def criterion(self,intermediates,target):
        loss =0
        minimum_length = min(intermediates[0].size(1),target.size(1))
        target = target[:,:minimum_length,:].clone()
        for intermediate in intermediates:
            intermediate = intermediate[:,:minimum_length,:].clone()
            loss = loss+ self.mae_loss(intermediate,target).clone()
            loss = loss+ self.mse_loss(intermediate,target).clone()
            loss = loss+ self.mse_loss(intermediate,target).clone() / (target.norm(p=2).pow(2))
        return loss
