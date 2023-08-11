from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
import torch

class MiipherDataModule(LightningDataModule):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
    def setup(self, stage:str):
        self.train_dataset = (
            wds.WebDataset(self.cfg.data.train_dataset_path).shuffle(1000).decode(wds.torch_audio)
        )
        self.val_dataset = (
            wds.WebDataset(self.cfg.data.val_dataset_path).shuffle(1000).decode(wds.torch_audio)
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=20
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=20
        )
    @torch.no_grad()
    def collate_fn(self, batch):
        output = dict()
        output['speaker'] = torch.stack([b['degraded_speaker_representation.pth'] for b in batch])
        output['phone'] = pad_sequence([b['phone_feature.pth'] for b in batch],batch_first=True)
        output['degraded_ssl'] = pad_sequence([b['degraded_ssl_feature.pth'] for b in batch],batch_first=True)
        output['clean_ssl'] = pad_sequence([b['clean_ssl_feature.pth'] for b in batch],batch_first=True)
        output['ssl_lengths'] = torch.tensor([b['degraded_ssl_feature.pth'].size(0) for b in batch])
        output['phone_lengths'] = torch.tensor([b['phone_feature.pth'].size(0) for b in batch])
        return output
