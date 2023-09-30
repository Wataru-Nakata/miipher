import gradio as gr
from miipher.dataset.preprocess_for_infer import PreprocessForInfer
from miipher.lightning_module import MiipherLightningModule
from lightning_vocoders.models.hifigan.xvector_lightning_module import HiFiGANXvectorLightningModule
import torch
import torchaudio
import hydra
import tempfile

miipher_path = "https://huggingface.co/spaces/Wataru/Miipher/resolve/main/miipher.ckpt"
miipher = MiipherLightningModule.load_from_checkpoint(miipher_path,map_location='cpu')
vocoder = HiFiGANXvectorLightningModule.load_from_checkpoint("https://huggingface.co/spaces/Wataru/Miipher/resolve/main/vocoder_finetuned.ckpt",map_location='cpu')
xvector_model = hydra.utils.instantiate(vocoder.cfg.data.xvector.model)
xvector_model = xvector_model.to('cpu')
preprocessor = PreprocessForInfer(miipher.cfg)

@torch.inference_mode()
def main(wav_path,transcript,lang_code):
    wav,sr =torchaudio.load(wav_path)
    wav = wav[0].unsqueeze(0)
    batch = preprocessor.process(
        'test',
        (torch.tensor(wav),sr),
        word_segmented_text=transcript,
        lang_code=lang_code
    )

    miipher.feature_extractor(batch)
    (
        phone_feature,
        speaker_feature,
        degraded_ssl_feature,
        _,
    ) = miipher.feature_extractor(batch)
    cleaned_ssl_feature, _ = miipher(phone_feature,speaker_feature,degraded_ssl_feature)
    vocoder_xvector = xvector_model.encode_batch(batch['degraded_wav_16k'].view(1,-1).cpu()).squeeze(1)
    cleaned_wav = vocoder.generator_forward({"input_feature": cleaned_ssl_feature, "xvector": vocoder_xvector})[0].T
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as fp:
        torchaudio.save(fp,cleaned_wav.view(1,-1), sample_rate=22050,format='wav')
        return fp.name

inputs = [gr.Audio(label="noisy audio",type='filepath'),gr.Textbox(label="Transcript", value="Your transcript here", max_lines=1), 
            gr.Radio(label="Language", choices=["eng-us", "jpn"], value="eng-us")]
outputs = gr.Audio(label="Output")

demo = gr.Interface(fn=main, inputs=inputs, outputs=outputs)

demo.launch()
