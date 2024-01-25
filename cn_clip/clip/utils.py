# Code modified from https://github.com/openai/CLIP

import json
import os
from pathlib import Path
from typing import Union, List
import urllib.request
from tqdm import tqdm
import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import ToTensor, Normalize, Resize, Inter

from . import _tokenizer
from .model import CLIP, convert_weights

__all__ = [
    "tokenize",
    "available_models",
    "image_transform",
    "load_from_name",
]

_MODELS = {
    "ViT-B-16": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-b-16.pt",
    "ViT-L-14": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14.pt",
    "ViT-L-14-336": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-l-14-336.pt",
    "ViT-H-14": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_vit-h-14.pt",
    "RN50": "https://clip-cn-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/clip_cn_rn50.pt",
}
_MODEL_INFO = {
    "ViT-B-16": {
        "struct": "ViT-B-16@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14": {
        "struct": "ViT-L-14@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 224
    },
    "ViT-L-14-336": {
        "struct": "ViT-L-14-336@RoBERTa-wwm-ext-base-chinese",
        "input_resolution": 336
    },
    "ViT-H-14": {
        "struct": "ViT-H-14@RoBERTa-wwm-ext-large-chinese",
        "input_resolution": 224
    },
    "RN50": {
        "struct": "RN50@RBT3-chinese",
        "input_resolution": 224
    },
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target) or os.path.isfile(download_target.replace(".pt", ".npy")) or os.path.isfile(download_target.replace(".pt", ".ckpt")):
        return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True,
                  unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_from_name(name: str, download_root: str = None, vision_model_name: str = None, text_model_name: str = None, input_resolution: int = None):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        model_name, model_input_resolution = _MODEL_INFO[name]['struct'], _MODEL_INFO[name]['input_resolution']
    elif os.path.isfile(name):
        assert vision_model_name and text_model_name and input_resolution, "Please specify specific 'vision_model_name', 'text_model_name', and 'input_resolution'"
        model_path = name
        model_name, model_input_resolution = f'{vision_model_name}@{text_model_name}', input_resolution
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    model = create_model(model_name, model_path)
    return model, image_transform(model_input_resolution)


def tokenize(texts: Union[str, List[str]], context_length: int = 52):
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all baseline models use 52 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    all_tokens = []
    for text in texts:
        all_tokens.append([_tokenizer.vocab['[CLS]']] + _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(text))[
                                                        :context_length - 2] + [_tokenizer.vocab['[SEP]']])

    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        assert len(tokens) <= context_length
        result[i, :len(tokens)] = np.array(tokens)

    return result


def _convert_to_rgb(image):
    return image.convert('RGB')


def image_transform(image_size=224):
    transform = Compose([
        Resize((image_size, image_size), interpolation=Inter.BICUBIC),
        _convert_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), is_hwc=False),
    ])
    return transform


def create_model(model_name, checkpoint_file=None):
    vision_model, text_model = model_name.split('@')
    # Initialize the model.
    vision_model_config_file = Path(
        __file__).parent / f"model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)

    text_model_config_file = Path(
        __file__).parent / f"model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)

    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        for k, v in json.load(ft).items():
            model_info[k] = v
    if isinstance(model_info['vision_layers'], str):
        model_info['vision_layers'] = eval(model_info['vision_layers'])
    print('Model info', model_info)
    model = CLIP(**model_info)
    # convert_weights(model)
    if checkpoint_file:
        def refiner(sd):
            if next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
            return sd
        load_pt_weights_in_model(model, checkpoint_file, (refiner,))

    return model


def get_pt2ms_mappings(model):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in model.cells_and_names():
        if isinstance(cell, nn.Conv1d):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: np.expand_dims(x, axis=-2)
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings


def convert_state_dict(model, state_dict_pt):
    mappings = get_pt2ms_mappings(model)
    state_dict_ms = {}
    for name_pt, data_pt in state_dict_pt.items():
        name_ms, data_mapping = mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = data_mapping(data_pt)
        if name_ms is not None:
            state_dict_ms[name_ms] = ms.Parameter(data_ms.astype(np.float32), name=name_ms)
    return state_dict_ms


def load_pt_weights_in_model(model, checkpoint_file_pt, state_dict_refiners=None):
    checkpoint_file_ms = f"{os.path.splitext(checkpoint_file_pt)[0]}.ckpt"
    if not os.path.exists(checkpoint_file_ms):  # try to load weights from intermediary numpy file.
        checkpoint_file_np = f"{os.path.splitext(checkpoint_file_pt)[0]}.npy"
        if not os.path.exists(checkpoint_file_np):
            raise FileNotFoundError(f"You need to manually convert {checkpoint_file_pt} to {checkpoint_file_np}")
        sd_original = np.load(checkpoint_file_np, allow_pickle=True).item()
        # refine state dict of pytorch
        sd_refined = sd_original
        if state_dict_refiners:
            for refine_fn in state_dict_refiners:
                sd_refined = refine_fn(sd_refined)
        # convert state_dict from pytorch to mindspore
        sd = convert_state_dict(model, sd_refined)
        # save converted state_dict as cache
        ms.save_checkpoint([{"name": k, "data": v} for k, v in sd.items()], checkpoint_file_ms)
    else:  # directly load weights from cached mindspore file.
        sd = ms.load_checkpoint(checkpoint_file_ms)

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, sd, strict_load=True)
    if param_not_load:
        print(f"{param_not_load} in network is not loaded!")
    if ckpt_not_load:
        print(f"{ckpt_not_load} in checkpoint is not loaded!")

