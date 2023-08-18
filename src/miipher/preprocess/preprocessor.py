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
from concurrent.futures import ThreadPoolExecutor, as_completed


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
        self.phoneme_tokenizer = hydra.utils.instantiate(cfg.preprocess.phoneme_tokenizer)
        self.degration_model = DegrationApplier(cfg.preprocess.degration)
        self.text2phone_dict = dict()
        self.n_repeats = cfg.preprocess.n_repeats
    @torch.inference_mode()
    def process_utterance(
        self,
        basename: str,
        audio_file_path,
        word_segmented_text: str,
        lang_code: str,
    ):
        orig_waveform, sample_rate = torchaudio.load(audio_file_path)

        waveform = torchaudio.functional.resample(
            orig_waveform, sample_rate, new_freq=self.sampling_rate
        )[
            0
        ]  # remove channel dimension only support mono

        with open(audio_file_path, mode="rb") as f:
            wav_bytes = f.read()


        input_ids, input_phonems = self.get_phonemes_input_ids(word_segmented_text,lang_code)

        samples = []
        for i in range(self.n_repeats):
            degraded_speech = self.apply_noise(waveform)
            buff = io.BytesIO()
            torchaudio.save(buff,src=degraded_speech.unsqueeze(0),sample_rate=self.sampling_rate,format='wav')
            buff.seek(0)

            
            sample = {
                "__key__": basename + f"_{i}",
                "speech.wav": wav_bytes,
                "degraded_speech.wav": buff.read(),
                "resampled_speech.pth": webdataset.torch_dumps(waveform),
                "word_segmented_text.txt": word_segmented_text,
                "phoneme_input_ids": webdataset.torch_dumps(input_ids),
                "phoneme.txt": input_phonems,
            }
            samples.append(sample)
        return samples

    def apply_noise(self,waveform):
        waveform = self.degration_model.process(waveform,self.sampling_rate)
        return waveform


    @torch.inference_mode()
    def get_phonemes_input_ids(self,word_segmented_text,lang_code):
        if lang_code not in self.text2phone_dict.keys():
            self.text2phone_dict[lang_code] = hydra.utils.instantiate(self.cfg.preprocess.text2phone_model,language=lang_code)
        input_phonemes = self.text2phone_dict[lang_code].infer_sentence(word_segmented_text)
        input_ids = self.phoneme_tokenizer(input_phonemes,return_tensors='pt')
        return input_ids, input_phonemes

    def build_from_path(self):
        pathlib.Path("/".join(self.cfg.preprocess.train_tar_sink.pattern.split("/")[:-1])).mkdir(exist_ok=True)
        train_sink = hydra.utils.instantiate(self.cfg.preprocess.train_tar_sink)
        val_sink = hydra.utils.instantiate(self.cfg.preprocess.val_tar_sink)
        dataloader = DataLoader(self.dataset,batch_size=1,shuffle=True,num_workers=64)
        for idx, data in enumerate(tqdm.tqdm(dataloader)):
            basename = data['basename'][0]
            wav_path = data['wav_path'][0]
            word_segmented_text = data['word_segmented_text'][0]
            lang_code = data['lang_code'][0]
            result = self.process_utterance(basename,wav_path,word_segmented_text,lang_code)
            if idx >= self.cfg.preprocess.val_size:
                sink = train_sink
            else:
                sink = val_sink
            for sample in result:
                sink.write(sample)
        train_sink.close()
        val_sink.close()
