from typing import Any, Optional
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from .model.miipher import Miipher
from omegaconf import DictConfig
from torch import nn
from typing import List
import torch
import hydra


class FeatureExtractor():
    def __init__(self,cfg) -> None:
        self.speech_ssl_model = hydra.utils.instantiate(cfg.model.ssl_models.model)
        self.speech_ssl_model.eval()
        self.phoneme_model = hydra.utils.instantiate(cfg.model.phoneme_model)
        self.phoneme_model.eval()
        self.xvector_model = hydra.utils.instantiate(cfg.model.xvector_model)
        self.xvector_model.eval()
        self.cfg = cfg

    @torch.inference_mode()
    def __call__(self, inputs):
        wav_16k = inputs["degraded_wav_16k"]
        wav_16k_lens = inputs["degraded_wav_16k_lengths"]
        feats = self.xvector_model.mods.compute_features(wav_16k)
        feats = self.xvector_model.mods.mean_var_norm(feats, wav_16k_lens)
        xvector = self.xvector_model.mods.embedding_model(feats, wav_16k_lens).squeeze(
            1
        )
        phone_feature = self.phoneme_model(
            **inputs["phoneme_input_ids"]
        ).last_hidden_state
        clean_ssl_feature = self.speech_ssl_model(
            **inputs["clean_ssl_input"], output_hidden_states=True
        )
        clean_ssl_feature = clean_ssl_feature.hidden_states[
            self.cfg.model.ssl_models.layer
        ]

        degraded_ssl_feature = self.speech_ssl_model(
            **inputs["degraded_ssl_input"], output_hidden_states=True
        )
        degraded_ssl_feature = degraded_ssl_feature.hidden_states[
            self.cfg.model.ssl_models.layer
        ]

        return phone_feature, xvector, degraded_ssl_feature, clean_ssl_feature

    def to(self,device:torch.device):
        print(device)
        self.speech_ssl_model = self.speech_ssl_model.to(device)
        self.phoneme_model = self.phoneme_model.to(device)
        self.xvector_model = self.xvector_model.to(device)

class MiipherLightningModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.miipher = Miipher(**cfg.model.miipher)
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cfg = cfg
        self.feature_extractor = FeatureExtractor(cfg)
        self.save_hyperparameters()

    def on_fit_start(self):
        self.feature_extractor.to(self.device)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            clean_ssl_feature,
        ) = self.feature_extractor(batch)

        cleaned_feature, intermediates = self.miipher.forward(
            phone_feature.clone(), speaker_feature.clone(), degraded_ssl_feature.clone()
        )
        loss = self.criterion(intermediates, clean_ssl_feature,log=True,stage='train')
        self.log("train/loss", loss, batch_size=phone_feature.size(0),prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        (
            phone_feature,
            speaker_feature,
            degraded_ssl_feature,
            clean_ssl_feature,
        ) = self.feature_extractor(batch)
        cleaned_feature, intermediates = self.miipher.forward(
            phone_feature, speaker_feature, degraded_ssl_feature
        )
        loss = self.criterion(intermediates, clean_ssl_feature,log=True,stage='val')
        self.log("val/loss", loss, batch_size=phone_feature.size(0))
        return loss

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.cfg.optimizers, params=self.miipher.parameters())

    def criterion(self, intermediates: List[torch.Tensor], target: torch.Tensor,log=False,stage='train'):
        loss = 0
        minimum_length = min(intermediates[0].size(1), target.size(1))
        target = target[:, :minimum_length, :].clone()
        for idx, intermediate in enumerate(intermediates):
            intermediate = intermediate[:, :minimum_length, :].clone()
            loss = loss + self.mae_loss(intermediate, target).clone()
            mae_loss = self.mae_loss(intermediate, target)
            mse_loss = self.mse_loss(intermediate, target)
            spectoral_loss = ( (intermediate - target).norm(p=2, dim=(1, 2)).pow(2) / (target.norm(p=2, dim=(1, 2)).pow(2))).mean()
            loss += mae_loss + mse_loss + spectoral_loss
            if log:
                self.log(f'{stage}/{idx}/mae_loss', mae_loss)
                self.log(f'{stage}/{idx}/mse_loss', mse_loss)
                self.log(f'{stage}/{idx}/spectoral_loss', spectoral_loss)

        return loss
