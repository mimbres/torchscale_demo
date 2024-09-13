# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Modified to apply "offload_to_cpu" by args.offload_to_cpu=True

from torchscale.architecture.decoder import Decoder, DecoderLayer
from torchscale.architecture.encoder import Encoder, EncoderLayer
from torchscale.component.dilated_attention import DilatedAttention
from fairscale.nn import checkpoint_wrapper, wrap


class LongNetDecoderLayer(DecoderLayer):

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


class LongNetDecoder(Decoder):

    def build_decoder_layer(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        layer = LongNetDecoderLayer(
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


class LongNetEncoderLayer(EncoderLayer):

    def build_self_attention(self, embed_dim, args):
        return DilatedAttention(
            args,
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )


class LongNetEncoder(Encoder):

    def build_encoder_layer(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        layer = LongNetEncoderLayer(
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
