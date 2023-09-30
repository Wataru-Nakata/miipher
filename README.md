# miipher
This repository proviedes unofficial implementation of speech restoration model Miipher.
Miipher is originally proposed by Koizumi et. al. [arxiv](https://arxiv.org/abs/2303.01664)
Please note that the model provided in this repository doesn't represent the performance of the original model proposed by Koizumi et. al. as this implementation differs in many ways from the paper.

# Installation
Install with pip. The installation is confirmed on Python 3.10.11
```python
pip install git+https://github.com/Wataru-Nakata/miipher
```

# Pretrained model
The pretrained model is trained on [LibriTTS-R](http://www.openslr.org/141/) and [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus),
and provided in **CC-BY-NC-2.0 license**.

The models are hosted on [huggingface](https://huggingface.co/spaces/Wataru/Miipher/)

To use pretrained model, please refere to `examples/demo.py`

# LICENSE
Code in this repo: MIT License

Weights on huggingface: CC-BY-NC-2.0 license

