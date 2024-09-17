# Modified to apply "offload_to_cpu" by args.offload_to_cpu=True
# pip install mamba-ssm[causal-conv1d]
from torchscale.architecture.decoder import Decoder, DecoderLayer
from torchscale.architecture.encoder import Encoder, EncoderLayer
from fairscale.nn import checkpoint_wrapper, wrap
from mamba_ssm import Mamba

class MambaWrapper(Mamba):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, query, key=None, value=None):
        """
        query: (B, L, D)
        key: Ignored
        value: Ignored

        Returns:
        The output of the Mamba forward pass with query as the input.
        """
        # Only pass query (hidden_states) to the Mamba forward method.
        hidden_states = query
        return super().forward(hidden_states)


class SambaLikeDecoderLayer(DecoderLayer):

    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )


class SambaLikeDecoder(Decoder):

    def build_decoder_layer(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        layer = SambaLikeDecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            if hasattr(args, "offload_to_cpu") and args.offload_to_cpu:
                layer = checkpoint_wrapper(layer, offload_to_cpu=True)
            else:
                layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer


class SambaLikeEncoderLayer(EncoderLayer):

    def build_self_attention(self, embed_dim, args):
        return MambaWrapper(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=embed_dim, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2    # Block expansion factor
        )
        # return DilatedAttention(
        #     args,
        #     embed_dim,
        #     args.encoder_attention_heads,
        #     dropout=args.attention_dropout,
        #     self_attention=True,
        #     encoder_decoder_attention=False,
        #     subln=args.subln,
        # )


class SambaLikeEncoder(Encoder):

    def build_encoder_layer(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        layer = SambaLikeEncoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            if hasattr(args, "offload_to_cpu") and args.offload_to_cpu:
                layer = checkpoint_wrapper(layer, offload_to_cpu=True)
            else:
                layer = checkpoint_wrapper(layer)
        if args.fsdp:
            layer = wrap(layer)
        return layer
