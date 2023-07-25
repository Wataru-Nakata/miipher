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
import io
import threading
from concurrent.futures import ThreadPoolExecutor


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
        self.sampling_rate = self.cfg.sample_rate
        self.ssl_models = hydra.utils.instantiate(cfg.preprocess.ssl_models)
        self.phoneme_model = hydra.utils.instantiate(cfg.preprocess.phoneme_model)
        self.phoneme_tokenizer = hydra.utils.instantiate(cfg.preprocess.phoneme_tokenizer)
        self.xvector_model = hydra.utils.instantiate(cfg.preprocess.xvector_model) 
        self.degration_model = DegrationApplier(cfg.preprocess.degration)
        self.text2phone_dict = dict()
        self.n_repeats = cfg.preprocess.n_repeats
    @torch.no_grad()
    def process_utterance(
        self,
        basename: str,
        orig_waveform: torch.Tensor,
        sample_rate: int,
        audio_file_path,
        word_segmented_text: str,
        lang_code: str,
        sink: webdataset.ShardWriter,
    ):

        waveform = torchaudio.functional.resample(
            orig_waveform, sample_rate, new_freq=self.sampling_rate
        )[
            0
        ]  # remove channel dimension only support mono

        with open(audio_file_path, mode="rb") as f:
            wav_bytes = f.read()

        # ssl model feature extraction
        clean_ssl_feature = self.extract_ssl_feature(orig_waveform,sample_rate)

        phone_feature = self.extract_phone_feature(word_segmented_text,lang_code)

        clean_speaker_representation = self.extract_speaker_representation(orig_waveform,sample_rate)
        for i in range(self.n_repeats):
            degraded_speech = self.apply_noise(waveform)
            degraded_ssl_feature = self.extract_ssl_feature(degraded_speech,self.sampling_rate)
            degraded_speaker_representation = self.extract_speaker_representation(degraded_speech,self.cfg.sample_rate)
            buff = io.BytesIO()
            torchaudio.save(buff,src=degraded_speech.unsqueeze(0),sample_rate=self.sampling_rate,format='wav')
            buff.seek(0)

            
            sample = {
                "__key__": basename + f"_{i}",
                "speech.wav": wav_bytes,
                "degraded_speech.wav": buff.read(),
                "resampled_speech.pth": webdataset.torch_dumps(waveform),
                "clean_ssl_feature.pth": webdataset.torch_dumps( clean_ssl_feature.last_hidden_state[0].cpu()),
                "degraded_ssl_feature.pth": webdataset.torch_dumps( degraded_ssl_feature.last_hidden_state[0].cpu()),
                "phone_feature.pth": webdataset.torch_dumps(phone_feature),
                "word_segmented_text.txt": word_segmented_text,
                "clean_speaker_representation.pth": clean_speaker_representation,
                "degraded_speaker_representation.pth":degraded_speaker_representation 
            }
            sink.write(sample)

    def apply_noise(self,waveform):
        waveform = self.degration_model.process(waveform,self.sampling_rate)
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
    
    @torch.no_grad()
    def extract_phone_feature(self,word_segmented_text,lang_code):
        if lang_code not in self.text2phone_dict.keys():
            self.text2phone_dict[lang_code] = hydra.utils.instantiate(self.cfg.preprocess.text2phone_model,language=lang_code)
        input_phonemes = self.text2phone_dict[lang_code].infer_sentence(word_segmented_text)
        input_ids = self.phoneme_tokenizer(input_phonemes,return_tensors='pt')
        features = self.phoneme_model(**input_ids)
        return features
    def extract_speaker_representation(self,waveform,sample_rate):
        xvector_model = self.xvector_model
        wav_tensor = torchaudio.functional.resample(
            waveform=waveform, orig_freq=sample_rate, new_freq=16_000
        )
        xvector = xvector_model.encode_batch(wav_tensor)
        return xvector


    def build_from_path(self):
        pathlib.Path("/".join(self.cfg.preprocess.train_tar_sink.pattern.split("/")[:-1])).mkdir(exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        dataloader = DataLoader(self.dataset,batch_size=1,shuffle=True,num_workers=8)
        for idx, data in enumerate(tqdm.tqdm(dataloader)):
            basename = data['basename'][0]
            wav_path = data['wav_path'][0]
            wav_tensor = data['wav_tensor'][0].float()
            sr = data['sr'][0]
            word_segmented_text = data['word_segmented_text'][0]
            lang_code = data['lang_code'][0]
            if idx >= self.cfg.preprocess.val_size:
                sink = train_sink
            else:
                sink = val_sink

            self.process_utterance(
                basename,
                wav_tensor,
                sr,
                wav_path,
                word_segmented_text,
                lang_code,
                sink,
            )
        train_sink.close()
        val_sink.close()
