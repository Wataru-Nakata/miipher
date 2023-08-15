from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import webdataset as wds
import torch
import torchdata
from torchdata.datapipes.iter import FileLister, FileOpener
from miipher.dataset.decoders import basic_decode, torch_audio

class MiipherDataModule(LightningDataModule):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
    def setup(self, stage:str):
        self.train_dataset = self.setup_datapipe(self.cfg.data.dataset_root,self.cfg.data.train_dataset_pattern)
        self.val_dataset = self.setup_datapipe(self.cfg.data.dataset_root,self.cfg.data.val_dataset_pattern)

    def setup_datapipe(self,root,pattern,shuffle=False):
        datapipe1 = FileLister(root,pattern)
        if shuffle:
            datapipe1 = datapipe1.shuffle()
        datapipe2 = FileOpener(datapipe1,mode='b')
        dataset = datapipe2.load_from_tar().webdataset().map(torch_audio).map(basic_decode)
        return dataset
    
    def train_dataloader(self):
        return wds.WebLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=20
        )

    def val_dataloader(self):
        return wds.WebLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=20
        )
    @torch.no_grad()
    def collate_fn(self, batch):
        output = dict()
        output['speaker'] = torch.stack([b['.degraded_speaker_representation.pth'] for b in batch])
        output['phone'] = pad_sequence([b['.phone_feature.pth'] for b in batch],batch_first=True)
        output['degraded_ssl'] = pad_sequence([b['.degraded_ssl_feature.pth'] for b in batch],batch_first=True)
        output['clean_ssl'] = pad_sequence([b['.clean_ssl_feature.pth'] for b in batch],batch_first=True)
        output['ssl_lengths'] = torch.tensor([b['.degraded_ssl_feature.pth'].size(0) for b in batch])
        output['phone_lengths'] = torch.tensor([b['.phone_feature.pth'].size(0) for b in batch])
        return output
