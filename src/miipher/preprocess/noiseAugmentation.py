from typing import Any
from torch import nn as nn
import torchaudio
import random
import pyroomacoustics as pra
import numpy as np
import torch


class DegrationApplier():
    def __init__(self,cfg) -> None:
        self.format_encoding_pairs = [
            {
                "format": "mp3", 
                "compression": 16
            },
            {
                "format": "mp3", 
                "compression": 32
            },
            {
                "format": "mp3", 
                "compression": 64
            },
            {
                "format": "mp3", 
                "compression": 128
            },
            {
                "format": "vorbis", 
                "compression": -1
            },
            {
                "format": "vorbis", 
                "compression": 0
            },
            {
                "format": "vorbis", 
                "compression": 1
            },
            {
                "format": "wav", 
                "encoding": "ALAW",
                "bits_per_sample": 8
            },
        ]

        self.reverb_conditions = {"reverbation_times": {
            "max": 0.5, 
            "min": 0.2
            }, 
            "room_xy": {
                "max": 10.0,
                "min": 2.0
            },
            "room_z": {
                "max": 5.0,
                "min": 2.0
            },
        }
    def applyCodec(self,waveform,sample_rate):
        if len(self.format_encoding_pairs) == 0:
            return waveform
        param =  random.choice(self.format_encoding_pairs)
        waveform = torchaudio.functional.apply_codec(
            waveform=waveform,
            sample_rate=sample_rate,
            **param
        )
        return waveform
    def applyReverb(self,waveform):
        xy_minmax = self.reverb_conditions.room_xy
        z_minmax = self.reverb_conditions.room_z
        x = random.uniform(xy_minmax.min,xy_minmax.max)
        y = random.uniform(xy_minmax.min,xy_minmax.max)
        z = random.uniform(z_minmax.min, z_minmax.max)
        corners = np.array([[0.0], [0,y], [x,y], [x,0]]).T
        room = pra.Room.from_corners(corners,**self.reverb_conditions.room_params)
        room.extrude(z)
        room.add_source(self.cfg.reverb_conditions.source_pos)
        room.add_microphone(self.cfg.reverb_conditions.mic_pos)

        room.compute_rir()
        rir = torch.tensor(room.rir[0][0])
        rir = rir/rir.norm(p=2)
        augmented = torchaudio.functional.fftconvolve(waveform,rir)
        return augmented
    def applyBackgroundNoise(self,waveform,sample_rate):
        snr_min,snr_max = self.background_noise.snr
        snr = random.uniform(snr_min,snr_max)

        noise_path = random.choice(self.noise_audio_paths)
        noise, noise_sr = torchaudio.load(noise_path)
        noise = torchaudio.functional.resample(noise,noise_sr,sample_rate)
        augmented = torchaudio.functional.add_noise(waveform=waveform,noise=noise,snr=snr)
        return augmented

    def process(self,waveform,sample_rate):
        waveform = self.applyBackgroundNoise(waveform,sample_rate)
        waveform = self.applyReverb(waveform,sample_rate)
        waveform = self.applyCodec(waveform,sample_rate)
    def __call__(self,waveform,sample_rate):
        return self.process(waveform,sample_rate)

