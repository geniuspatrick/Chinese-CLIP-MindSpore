# -*- coding: utf-8 -*-
'''
This script extracts image and text features for evaluation. (with single-GPU)
'''

import os
import argparse
from pathlib import Path
import json
from tqdm import tqdm

import mindspore as ms

from cn_clip.clip.model import CLIP
from cn_clip.eval.data import get_eval_img_dataset, get_eval_txt_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--extract-image-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract image features."
    )
    parser.add_argument(
        '--extract-text-feats', 
        action="store_true", 
        default=False, 
        help="Whether to extract text features."
    )
    parser.add_argument(
        '--image-data', 
        type=str, 
        default="../Multimodal_Retrieval/lmdb/test/imgs", 
        help="If --extract-image-feats is True, specify the path of the LMDB directory storing input image base64 strings."
    )
    parser.add_argument(
        '--text-data', 
        type=str, 
        default="../Multimodal_Retrieval/test_texts.jsonl", 
        help="If --extract-text-feats is True, specify the path of input text Jsonl file."
    )
    parser.add_argument(
        '--image-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output image features."
    )    
    parser.add_argument(
        '--text-feat-output-path', 
        type=str, 
        default=None, 
        help="If --extract-image-feats is True, specify the path of output text features."
    )
    parser.add_argument(
        "--img-batch-size", type=int, default=64, help="Image batch size."
    )
    parser.add_argument(
        "--text-batch-size", type=int, default=64, help="Text batch size."
    )
    parser.add_argument(
        "--context-length", type=int, default=52, help="The maximum length of input text (include [CLS] & [SEP] tokens)."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--vision-model",
        choices=["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-L-14-336", "ViT-H-14", "RN50"],
        default="ViT-B-16",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--text-model",
        choices=["RoBERTa-wwm-ext-base-chinese", "RoBERTa-wwm-ext-large-chinese", "RBT3-chinese"],
        default="RoBERTa-wwm-ext-base-chinese",
        help="Name of the text backbone to use.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )    
    args = parser.parse_args()

    return args


def load_state_dict(checkpoint_path: str):
    checkpoint = ms.load_checkpoint(checkpoint_path)
    state_dict = checkpoint
    # TODO: We may should do unwrap stuffs like cleaning the augly "_backbone" prefix when saving checkpoints.
    if "optimizer.global_step" in state_dict:  # need to unwrap trainer-format checkpoint
        optimizer_state_dict = {}
        model_state_dict = {}
        mics_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("optimizer."):
                k = k[len("optimizer.") :]
                if k.startswith("adam") or k == "global_step" or "learning_rate" in k:
                    optimizer_state_dict[k] = v
                else:
                    model_state_dict[k] = v
            elif k.startswith("network."):
                k = k[len("network.") :]
                model_state_dict[k.replace("_backbone.", "")] = v
            else:
                mics_state_dict[k] = v
        state_dict = model_state_dict
    return state_dict


if __name__ == "__main__":
    args = parse_args()

    assert args.extract_image_feats or args.extract_text_feats, "--extract-image-feats and --extract-text-feats cannot both be False!"

    # Log params.
    print("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        print(f"  {name}: {val}")

    # Initialize the model.
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{args.text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        for k, v in json.load(ft).items():
            model_info[k] = v
    model = CLIP(**model_info)
    
    # Resume from a checkpoint.
    print("Begin to load model checkpoint from {}.".format(args.resume))
    assert os.path.exists(args.resume), "The checkpoint file {} not exists!".format(args.resume)
    state_dict = load_state_dict(args.resume)
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict, strict_load=True)
    if param_not_load:
        print(f"{param_not_load} in network is not loaded!")
    if ckpt_not_load:
        print(f"{ckpt_not_load} in checkpoint is not loaded!")

    # Make inference for texts
    if args.extract_text_feats:
        print("Preparing text inference dataset.")
        text_data = get_eval_txt_dataset(args, max_txt_length=args.context_length)
        print('Make inference for texts...')
        if args.text_feat_output_path is None:
            args.text_feat_output_path = "{}.txt_feat.jsonl".format(args.text_data[:-6])
        write_cnt = 0
        with open(args.text_feat_output_path, "w") as fout:
            model.set_train(False)
            dataloader = text_data.dataloader
            for batch in tqdm(dataloader):
                text_ids, texts = batch
                text_features = model(None, texts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                for text_id, text_feature in zip(text_ids.tolist(), text_features.tolist()):
                    fout.write("{}\n".format(json.dumps({"text_id": text_id, "feature": text_feature})))
                    write_cnt += 1
        print('{} text features are stored in {}'.format(write_cnt, args.text_feat_output_path))

    # Make inference for images
    if args.extract_image_feats:
        print("Preparing image inference dataset.")
        img_data = get_eval_img_dataset(args)
        print('Make inference for images...')
        if args.image_feat_output_path is None:
            # by default, we store the image features under the same directory with the text features
            args.image_feat_output_path = "{}.img_feat.jsonl".format(args.text_data.replace("_texts.jsonl", "_imgs"))
        write_cnt = 0
        with open(args.image_feat_output_path, "w") as fout:
            model.set_train(False)
            dataloader = img_data.dataloader
            for batch in tqdm(dataloader):
                image_ids, images = batch
                image_features = model(images, None)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                for image_id, image_feature in zip(image_ids.tolist(), image_features.tolist()):
                    fout.write("{}\n".format(json.dumps({"image_id": image_id, "feature": image_feature})))
                    write_cnt += 1
        print('{} image features are stored in {}'.format(write_cnt, args.image_feat_output_path))

    print("Done!")
