import torch
from torchscale.architecture.config import EncoderConfig, DecoderConfig
from torchscale.model.LongNet import LongNetEncoder, LongNetDecoder


# requirements: fairscale==0.4.0, timm==0.6.13, einops, flash_attention (optional) pip install flash-attn --no-build-isolation
def display_cfg(cfg):
    for k, v in vars(cfg).items():
        print(f'{k}: {v}')


def count_parameters(model):
    cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num of parameters: {cnt / 1_000_000:.0f}M")


# Creating a LongNet encoder with the dilated pattern of segment_length=[2048,4096] and dilated_ratio=[1,2]
enc_cfg = EncoderConfig(
    encoder_embed_dim=768,  #512,
    encoder_attention_heads=12,  #8, #12,
    encoder_ffn_embed_dim=3072,  #1024,
    encoder_layers=8,
    vocab_size=3000,
    segment_length='[4, 4]',  #'[2048,4096]',
    dilated_ratio='[1, 1]',  #'[1,2]',
    subln=True,
    dropout=0.05,
    flash_attention=False)
enc = LongNetEncoder(enc_cfg)
display_cfg(enc_cfg)
count_parameters(enc)

enc.embed_tokens = torch.nn.Embedding(3000, 768)
x = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
y = enc.forward(x)

x = torch.rand(2, 2048, 768)
y = enc.forward(src_tokens=None, token_embeddings=x)

# Creating a LongNet decoder with the dilated pattern of segment_length=[2048,4096] and dilated_ratio=[1,2]
dec_cfg = DecoderConfig(vocab_size=64000, segment_length='[2048,4096]', dilated_ratio='[1,2]', flash_attention=False)
dec = LongNetDecoder(dec_cfg)
display_cfg(dec_cfg)
count_parameters(dec)
