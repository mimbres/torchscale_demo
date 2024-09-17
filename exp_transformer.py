import torch
from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder
import torch.profiler

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32


def display_cfg(cfg):
    for k, v in vars(cfg).items():
        print(f'{k}: {v}')


def count_parameters(model):
    cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Num of parameters: {cnt / 1_000_000:.0f}M")


# Configurations for LongNetEncoder
enc_cfg = EncoderConfig(
    encoder_embed_dim=512,
    encoder_attention_heads=8,
    encoder_ffn_embed_dim=1024,
    encoder_layers=8,
    vocab_size=3000,
    subln=True,
    dropout=0.05,
    flash_attention=True,
    checkpoint_activations=True,  # gradient checkpointing
    offload_to_cpu=False,
)
enc = Encoder(enc_cfg).to(device, dtype)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                            profile_memory=True,
                            record_shapes=True,
                            with_stack=True) as prof:
    x = torch.rand(2, 20480 *4, 512, requires_grad=True).to(device, dtype)
    y = enc.forward(src_tokens=None, token_embeddings=x)["encoder_out"]  # Adjusted output access for LongNetEncoder
    target = torch.rand_like(y)

    loss = torch.nn.MSELoss()(y, target)
    loss.backward()

    optimizer = torch.optim.Adam(enc.parameters(), lr=1e-4)
    optimizer.step()

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
print(torch.cuda.memory_summary())
print(f"Loss: {loss.item()}")

# torch.cuda.empty_cache()  # Clear GPU memory
