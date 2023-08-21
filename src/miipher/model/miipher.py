from torch import nn
from .modules import FiLMLayer, PositionalEncoding, Postnet
from torchaudio.models.conformer import ConformerLayer
import torch


class Miipher(nn.Module):
    def __init__(
        self,
        n_phone_feature,
        n_speaker_embedding,
        n_ssl_feature,
        n_hidden_dim,
        n_conformer_blocks,
        n_iters,
    ) -> None:
        super().__init__()
        self.phone_speaker_film = FiLMLayer(n_hidden_dim, n_hidden_dim)
        self.phone_linear = nn.Linear(n_phone_feature, n_hidden_dim)
        self.speaker_linear = nn.Linear(n_speaker_embedding, n_hidden_dim)

        self.ssl_linear = nn.Linear(n_ssl_feature, n_hidden_dim)

        self.positional_encoding = PositionalEncoding(n_hidden_dim)
        self.positional_encoding_film = FiLMLayer(n_hidden_dim, n_hidden_dim)
        self.conformer_blocks = nn.ModuleList()
        for i in range(n_conformer_blocks):
            self.conformer_blocks.append(FeatureCleanerBlock(n_hidden_dim, 8))
        self.postnet = Postnet(
            n_hidden_dim,
            postnet_embedding_dim=512,
            postnet_kernel_size=5,
            postnet_n_convolutions=5,
        )
        self.n_iters = n_iters
        self.n_conformer_blocks = n_conformer_blocks

    def forward(
        self, phone_feature, speaker_feature, ssl_feature, ssl_feature_lengths=None
    ):
        """
        Args:
            phone_feature: (N, T, n_phone_feature)
            speaker_feature: (N, n_speaker_embedding)
            ssl_feature: (N, T, n_ssl_feature)
        """
        N = phone_feature.size(0)
        assert speaker_feature.size(0) == N
        assert ssl_feature.size(0) == N
        phone_feature = self.phone_linear(phone_feature)
        speaker_feature = self.speaker_linear(speaker_feature)
        ssl_feature = self.ssl_linear(ssl_feature)
        intermediates = []
        phone_speaker_feature = self.phone_speaker_film(
            phone_feature, speaker_feature.unsqueeze(1)
        )
        for iteration_count in range(self.n_iters):
            pos_enc = self.positional_encoding(
                torch.tensor(iteration_count, device=self.device).unsqueeze(0).repeat(N)
            )
            assert pos_enc.size(0) == N
            phone_speaker_feature = self.positional_encoding_film(
                phone_speaker_feature, pos_enc
            )
            for i in range(self.n_conformer_blocks):
                ssl_feature = self.conformer_blocks[i](
                    ssl_feature.clone(), phone_speaker_feature, ssl_feature_lengths
                )
            intermediates.append(ssl_feature.clone())
            ssl_feature += self.postnet(ssl_feature.clone())
            intermediates.append(ssl_feature.clone())
        return ssl_feature, torch.stack(intermediates)

    @property
    def device(self):
        return next(iter(self.parameters())).device


class FeatureCleanerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads) -> None:
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.conformer_block = ConformerLayer(hidden_dim, hidden_dim * 4, num_heads, 31)
        self.layer_norm = nn.LayerNorm(1024)

    def forward(
        self, cleaning_feature, speaker_phone_feature, cleaning_feature_lengths=None
    ):
        if cleaning_feature_lengths is not None:
            mask = _lengths_to_padding_mask(cleaning_feature_lengths).T
        else:
            mask = None
        cleaning_feature += self.cross_attention(
            cleaning_feature.clone(),
            speaker_phone_feature,
            speaker_phone_feature,
        )[0]
        cleaning_feature = self.layer_norm(cleaning_feature.clone())
        cleaning_feature += self.conformer_block(
            cleaning_feature.clone(), key_padding_mask=mask
        )
        return cleaning_feature


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(
        max_length, device=lengths.device, dtype=lengths.dtype
    ).expand(batch_size, max_length) >= lengths.unsqueeze(1)
    return padding_mask
