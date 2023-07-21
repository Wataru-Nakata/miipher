import torch
import hydra
import torchaudio
import pathlib
from omegaconf import DictConfig
import numpy as np
import webdataset
import tqdm
from torch.utils.data import DataLoader
from speechbrain.pretrained import EncoderClassifier
from .noiseAugmentation import DegrationApplier


class Preprocessor:
    """
    Preprocess dataset
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: hydra config
        """
        self.cfg = cfg
        self.dataset = hydra.utils.instantiate(cfg.preprocess.preprocess_dataset)
        self.spec_module = torchaudio.transforms.Spectrogram(**cfg.preprocess.stft)
        self.mel_scale = torchaudio.transforms.MelScale(**cfg.preprocess.mel)
        self.sampling_rate = self.cfg.sample_rate
        self.ssl_models = hydra.utils.instantiate(cfg.preprocess.ssl_models)
        self.phoneme_model = hydra.utils.instantiate(cfg.preprocess.phoneme_model)
        self.xvector_model = hydra.utils.instantiate(cfg.preprocess.xvector_model) 
        self.degration_model = DegrationApplier(cfg.)

    @torch.no_grad()
    def process_utterance(
        self,
        basename: str,
        orig_waveform: torch.Tensor,
        sample_rate: int,
        audio_file_path,
        word_segmented_text: str
    ):

        waveform = torchaudio.functional.resample(
            orig_waveform, sample_rate, new_freq=self.sampling_rate
        )[
            0
        ]  # remove channel dimension only support mono

        mel_spec, _ = self.calc_spectrogram(waveform)
        with open(audio_file_path, mode="rb") as f:
            wav_bytes = f.read()

        # ssl model feature extraction
        ssl_feature = self.extract_ssl_feature(orig_waveform,sample_rate)

        phone_feature = self.extract_phone_feature(word_segmented_text)

        speaker_representation = self.extract_speaker_representation(orig_waveform,sample_rate)

        sample = {
            "__key__": basename,
            "speech.wav": wav_bytes,
            "resampled_speech.pth": webdataset.torch_dumps(waveform),
            "mel.pth": webdataset.torch_dumps(mel_spec.T),
            "ssl_feature.pth": webdataset.torch_dumps( ssl_feature.last_hidden_state[0].cpu()),
            "phone_feature.pth": phone_feature,
            "word_segmented_text.txt": word_segmented_text,
            "speaker_representation.pth": speaker_representation
        }

        return sample
    def apply_noise(self,waveform):
        for noise_func in self.noise_funcs:
            waveform = noise_func(waveform)
        return waveform

    def extract_ssl_feature(self,waveform:torch.Tensor,sample_rate):
        ssl_model, processor, feature_cfg = self.ssl_models
        wav_tensor = torchaudio.functional.resample(
            waveform=waveform, orig_freq=sample_rate, new_freq=feature_cfg.sr
        )
        inputs = processor(
            wav_tensor.squeeze(), return_tensors="pt", sampling_rate=feature_cfg.sr
        )
        inputs.to("cuda")
        ssl_model.to("cuda")
        ssl_feature = ssl_model(**inputs)
        return ssl_feature
    
    def extract_phone_feature(self,text):
        raise NotImplementedError
    def extract_speaker_representation(self,waveform,sample_rate):
        xvector_model, xvector_cfg = self.xvector_model
        wav_tensor = torchaudio.functional.resample(
            waveform=waveform, orig_freq=sample_rate, new_freq=xvector_cfg.sr
        )
        xvector = xvector_model.encode_batch(wav_tensor)
        return xvector


    def build_from_path(self):
        pathlib.Path("/".join(self.cfg.preprocess.train_tar_sink.pattern.split("/")[:-1])).mkdir(exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        dataloader = DataLoader(self.dataset,batch_size=1)
        for idx, (basename, (wav,sr),wav_path,word_segmented_text) in enumerate(tqdm.tqdm(dataloader)):
            sample = self.process_utterance(
                basename[0],
                wav[0],
                sr[0],
                wav_path[0],
                word_segmented_text[0]
            )
            if idx >= self.cfg.preprocess.val_size:
                train_sink.write(sample)
            else:
                val_sink.write(sample)

        train_sink.close()
        val_sink.close()

    def calc_spectrogram(self, waveform: torch.Tensor):
        magspec = self.spec_module(waveform)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec, energy.numpy()
