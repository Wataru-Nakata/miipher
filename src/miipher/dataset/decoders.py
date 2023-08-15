from torchdata.datapipes.utils import StreamWrapper
import torchaudio
import pickle
import json
import re
import torch

def basic_decode(sample_dict):
    for key, value in sample_dict.items():
            sample_dict[key]= basichandlers(key,value)
    return sample_dict
def basichandlers(key, data):
    """Handle basic file decoding.

    This function is usually part of the post= decoders.
    This handles the following forms of decoding:

    - txt -> unicode string
    - cls cls2 class count index inx id -> int
    - json jsn -> JSON decoding
    - pyd pickle -> pickle decoding
    - pth -> torch.loads
    - ten tenbin -> fast tensor loading
    - mp messagepack msg -> messagepack decoding
    - npy -> Python NPY decoding

    :param key: file name extension
    :param data: binary data to be decoded
    """
    extension = re.sub(r".*[.]", "", key)

    if extension in "txt text transcript":
        return data.read().decode("utf-8")

    if extension in "cls cls2 class count index inx id".split():
        try:
            return int(data.read())
        except ValueError:
            return None

    if extension in "json jsn":
        return json.loads(data.read())

    if extension in "pyd pickle".split():
        return pickle.loads(data)

    if extension in "pth".split():
        return torch.load(data)
    return data

def torch_audio(sample_dict):
    for key, value in sample_dict.items():
        if key.endswith(".wav"):
            sample_dict[key]= torchaudio.load(value)
    return sample_dict
