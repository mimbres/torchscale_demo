import torch
from torchscale.architecture.config import EncoderConfig
from mambapy.mamba import Mamba, MambaConfig
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
enc_cfg = MambaConfig(d_model=512, n_layers=8, use_cuda=bool(device == "cuda"))
enc = Mamba(enc_cfg).to(device, dtype)
enc_compile = torch.compile(enc, mode="reduce-overhead")

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                            profile_memory=True,
                            record_shapes=True,
                            with_stack=True) as prof:
    x = torch.rand(2, 20480 * 4, 512, requires_grad=True).to(device, dtype)
    y = enc_compile.forward(x)  # Adjusted output access for LongNetEncoder
    target = torch.rand_like(y)

    loss = torch.nn.MSELoss()(y, target)
    loss.backward()

    optimizer = torch.optim.Adam(enc.parameters(), lr=1e-4)
    optimizer.step()

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
print(torch.cuda.memory_summary())
print(f"Loss: {loss.item()}")

# torch.cuda.empty_cache()  # Clear GPU memory
