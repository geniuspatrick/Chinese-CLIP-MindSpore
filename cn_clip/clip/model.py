from collections import OrderedDict
from typing import Tuple, Union
from itertools import repeat
import collections.abc
import numbers

import math
import logging
import numpy as np

import mindspore as ms
from mindspore import ops, nn, Tensor, Parameter
from .layer import multi_head_attention_forward, MultiheadAttention

from . import _tokenizer
from .configuration_bert import BertConfig
from .modeling_bert import BertModel, normal_, zeros_

_logger = logging.getLogger(__name__)


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(planes, planes, 3, pad_mode="pad", padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU()

        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.SequentialCell(OrderedDict([
                ("-1", nn.AvgPool2d(stride, stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, has_bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Cell):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = Parameter(ops.randn((spacial_dim**2 + 1, embed_dim)) / embed_dim**0.5)
        self.k_proj = nn.Dense(embed_dim, embed_dim)
        self.q_proj = nn.Dense(embed_dim, embed_dim)
        self.v_proj = nn.Dense(embed_dim, embed_dim)
        self.c_proj = nn.Dense(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def construct(self, x):
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3])).permute((2, 0, 1))  # NCHW -> (HW)NC
        x = ops.cat([x.mean(axis=0, keep_dims=True), x], axis=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=ops.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
        )

        return x[0]


class ModifiedResNet(nn.Cell):
    """
    A ResNet class that contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, pad_mode="pad", padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, pad_mode="pad", padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, pad_mode="pad", padding=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2, 2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        # todo: self.apply(reset_parameters_torch)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.SequentialCell(*layers)

    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def construct(self, x, mask_ratio: float = 0.0):
        assert mask_ratio == 0.0, "mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer."
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def __init__(self, normalized_shape, eps=1e-5, dtype=ms.float32):
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        super().__init__(normalized_shape, epsilon=eps, dtype=dtype)

    def construct(self, x: Tensor):
        orig_type = x.dtype
        x, _, _ = self.layer_norm(x.to(ms.float32), self.gamma.to(ms.float32), self.beta.to(ms.float32))
        return x.to(orig_type)


class GELU(nn.GELU):
    def __init__(self, approximate: str = "none"):
        if approximate == "none":
            super().__init__(False)
        elif approximate == "tanh":
            super().__init__(True)
        else:
            raise ValueError(f"approximate must be one of ['none', 'tanh'], but got {approximate}.")


class QuickGELU(nn.Cell):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model: int, n_head: int, attn_mask: Tensor = None, use_flash_attention: bool = False):
        super().__init__()

        assert use_flash_attention is False
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.SequentialCell(OrderedDict([
            ("c_fc", nn.Dense(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Dense(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_flash_attention = use_flash_attention

    def attention(self, x: Tensor):
        attn_mask = self.attn_mask.to(dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def construct(self, x: Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Tensor = None, use_flash_attention: bool = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.resblocks = nn.SequentialCell(*[ResidualAttentionBlock(width, heads, attn_mask, use_flash_attention) for _ in range(layers)])

    def construct(self, x: Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Cell):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_flash_attention: bool = False):
        super().__init__()
        self.input_resolution = input_resolution
        self.grid_size = (self.input_resolution // patch_size, self.input_resolution // patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, has_bias=False)

        scale = width ** -0.5
        self.class_embedding = Parameter(scale * ops.randn(width))
        self.positional_embedding = Parameter(scale * ops.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, use_flash_attention=use_flash_attention)

        self.ln_post = LayerNorm(width)
        self.proj = Parameter(scale * ops.randn(width, output_dim))

    def set_grad_checkpointing(self, enable=True):
        _logger.warning("Gradient checkpointing is not supported. Setting enable is omitted.")
        self.transformer.grad_checkpointing = enable

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int((L - 1) * (1 - mask_ratio))

        noise = ops.rand(N, L - 1)
        ids_shuffle = ops.argsort(noise, axis=1) + ops.ones((N, L - 1), dtype=ms.int32)
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = ops.gather(x, axis=1, input_indices=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        x0 = x[:, 0, :]
        x0 = x0.reshape(N, 1, D)
        x_masked_add = ops.cat([x0, x_masked], axis=1)
        return x_masked_add

    def construct(self, x: Tensor, mask_ratio: float = 0.0):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = ops.cat([self.class_embedding.to(x.dtype) + ops.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype), x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if mask_ratio != 0:
            x = self.random_masking(x, mask_ratio)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj.to(x.dtype)

        return x


class CLIP(nn.Cell):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 vocab_size: int,
                 text_attention_probs_dropout_prob: float, 
                 text_hidden_act: str, 
                 text_hidden_dropout_prob: float, 
                 text_hidden_size: int,
                 text_initializer_range: float, 
                 text_intermediate_size: int, 
                 text_max_position_embeddings: int, 
                 text_num_attention_heads: int, 
                 text_num_hidden_layers: int, 
                 text_type_vocab_size: int,
                 tokenizer = _tokenizer,
                 # vision head width, added this param for ViT-H
                 vision_head_width: int = 64,
                 use_flash_attention: bool = False,
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // vision_head_width
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // vision_head_width
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_flash_attention=use_flash_attention
            )

        self.bert_config = BertConfig(
            vocab_size_or_config_json_file=vocab_size,
            hidden_size=text_hidden_size,
            num_hidden_layers=text_num_hidden_layers,
            num_attention_heads=text_num_attention_heads,
            intermediate_size=text_intermediate_size,
            hidden_act=text_hidden_act,
            hidden_dropout_prob=text_hidden_dropout_prob,
            attention_probs_dropout_prob=text_attention_probs_dropout_prob,
            max_position_embeddings=text_max_position_embeddings,
            type_vocab_size=text_type_vocab_size,
            initializer_range=text_initializer_range,
            layer_norm_eps=1e-12,
            use_flash_attention=use_flash_attention
        )
        self.bert = BertModel(self.bert_config)

        self.text_projection = Parameter(ops.randn(text_hidden_size, embed_dim))
        self.logit_scale = Parameter(ops.ones(()) * np.log(1 / 0.07))

        self.tokenizer = tokenizer
        self.pad_index = self.tokenizer.vocab['[PAD]']  # int: 0

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_channels ** -0.5
                normal_(self.visual.attnpool.q_proj.weight, std=std)
                normal_(self.visual.attnpool.k_proj.weight, std=std)
                normal_(self.visual.attnpool.v_proj.weight, std=std)
                normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.parameters_and_names():
                    if name.endswith("bn3.weight"):
                        zeros_(param)

        if self.text_projection is not None:
            normal_(self.text_projection, std=self.bert_config.hidden_size ** -0.5)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.bert.set_grad_checkpointing(enable)

    def encode_image(self, image, mask_ratio: float = 0.0):
        return self.visual(image.type(self.visual.conv1.weight.dtype), mask_ratio)

    def encode_text(self, text):
        attn_mask = text.ne(self.pad_index)
        x = self.bert(text, attention_mask=attn_mask)[0]  # [batch_size, seq_length, hidden_size]
        return x[:, 0, :] @ self.text_projection.to(x.dtype)

    def construct(self, image, text, mask_ratio: float = 0.0):
        assert image is not None or text is not None, "text and image cannot both be None!"

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image, mask_ratio)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale.exp()

    def get_similarity(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Cell):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Dense)):
            l.weight.set_dtype(ms.float16)
            if l.bias is not None:
                l.bias.set_dtype(ms.float16)

        if isinstance(l, MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.set_dtype(ms.float16)

        if isinstance(l, BertModel):
            for p in l.get_parameters():
                p.set_dtype(ms.float16)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.set_dtype(ms.float16)

    model.apply(_convert_weights_to_fp16)


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', seq_dim=1, prefix=""):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get(prefix + 'visual.positional_embedding', None)
    model = model.module if hasattr(model, 'module') else model
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = ops.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        align_corners=True,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = ops.cat([pos_emb_tok, pos_emb_img], axis=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict[prefix + 'visual.positional_embedding'] = new_pos_embed


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)
