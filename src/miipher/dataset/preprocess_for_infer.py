import torch 
import hydra
import torchaudio
from torch.nn.utils.rnn import pad_sequence

class PreprocessForInfer(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.phoneme_tokenizer = hydra.utils.instantiate(
            cfg.preprocess.phoneme_tokenizer
        )
        self.speech_ssl_processor = hydra.utils.instantiate(
            cfg.data.speech_ssl_processor.processor
        )
        self.speech_ssl_sr = cfg.data.speech_ssl_processor.sr
        self.cfg = cfg
        self.text2phone_dict = dict()

    @torch.inference_mode()
    def get_phonemes_input_ids(self, word_segmented_text, lang_code):
        if lang_code not in self.text2phone_dict.keys():
            self.text2phone_dict[lang_code] = hydra.utils.instantiate(
                self.cfg.preprocess.text2phone_model, language=lang_code
            )
        input_phonemes = self.text2phone_dict[lang_code].infer_sentence(
            word_segmented_text
        )
        input_ids = self.phoneme_tokenizer(input_phonemes, return_tensors="pt")
        return input_ids, input_phonemes
    def process(self,basename, degraded_audio,word_segmented_text=None,lang_code=None, phoneme_text=None):
        degraded_audio,sr = degraded_audio
        output = dict()

        if word_segmented_text != None and  lang_code != None:
            input_ids, input_phonems = self.get_phonemes_input_ids(
                word_segmented_text, lang_code
            )
            output['phoneme_input_ids'] = input_ids
        elif phoneme_text == None:
            raise ValueError
        else:
            output["phoneme_input_ids"] = self.phoneme_tokenizer(
                phoneme_text, return_tensors="pt", padding=True
            )

        degraded_16k = torchaudio.functional.resample(
            degraded_audio, sr, new_freq=16000
        ).squeeze()
        degraded_wav_16ks = [degraded_16k]

        output["degraded_ssl_input"] = self.speech_ssl_processor(
            [x.numpy() for x in degraded_wav_16ks],
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        output["degraded_wav_16k"] = pad_sequence(degraded_wav_16ks, batch_first=True)
        output["degraded_wav_16k_lengths"] = torch.tensor(
            [degraded_wav_16k.size(0) for degraded_wav_16k in degraded_wav_16ks]
        )
        return output

