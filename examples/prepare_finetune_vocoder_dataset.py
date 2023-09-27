import torch
from torch.utils.data import DataLoader
from miipher.lightning_module import MiipherLightningModule
import webdataset
from pathlib import Path
from miipher.dataset.preprocess_for_infer import PreprocessForInfer
from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule
import hydra
from tqdm import tqdm

@torch.inference_mode()
def main(miipher_path: Path):
    torch.set_float32_matmul_precision("medium")
    miipher = MiipherLightningModule.load_from_checkpoint(miipher_path)
    train_dataset = webdataset.WebDataset(miipher.cfg.data.train_dataset_path).decode(webdataset.torch_audio)
    val_dataset = webdataset.WebDataset(miipher.cfg.data.val_dataset_path).decode(webdataset.torch_audio)
    train_dl = DataLoader(train_dataset,num_workers=8)
    val_dl = DataLoader(val_dataset,num_workers=8)
    preprocessor = PreprocessForInfer(miipher.cfg)
    train_sink = webdataset.TarWriter("/mnt/hdd/finetune_train.tar.gz")
    val_sink = webdataset.TarWriter("/mnt/hdd/finetune_val.tar.gz")
    device = miipher.device
    miipher.feature_extractor.to(device)
    vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint("https://huggingface.co/Wataru/ssl-vocoder/resolve/main/wavlm-large-l8-xvector/wavlm-large-l8-xvector.ckpt",map_location='cpu')
    vocoder.eval()
    xvector_model = hydra.utils.instantiate(vocoder.cfg.data.xvector.model)
    xvector_model = xvector_model.to('cpu')
    del(vocoder)
    xvector_model.eval()

    for dl,sink in zip([val_dl,train_dl],[val_sink,train_sink]):
        for sample in tqdm(dl):
            batch = preprocessor.process(
                sample['__key__'],
                sample['degraded_speech.wav'],
                phoneme_text= sample['phoneme.txt']
            )
            for k,v in batch.items():
                batch[k] = v.to(device)
            (
                phone_feature,
                speaker_feature,
                degraded_ssl_feature,
                _,
            ) = miipher.feature_extractor(batch)
            cleaned_ssl_feature, _ = miipher(phone_feature.to(device),speaker_feature.to(device),degraded_ssl_feature.to(device))
            vocoder_xvector = xvector_model.encode_batch(batch['degraded_wav_16k'].view(1,-1).cpu()).squeeze(1)

            sample_to_write = {
                "__key__": sample['__key__'][0],
                "resampled_speech.pth": webdataset.torch_dumps(sample['resampled_speech.pth'][0].cpu()),
                "miipher_cleaned_feature.pth": webdataset.torch_dumps(cleaned_ssl_feature[0].cpu()),
                "xvector.pth": webdataset.torch_dumps(vocoder_xvector.view(-1).cpu())
            }
            sink.write(sample_to_write)
        sink.close()



if __name__ == "__main__":
    ckpt_path = Path("miipher/0kt6hnn2/checkpoints/epoch=19-step=400000.ckpt")
    main(ckpt_path)
