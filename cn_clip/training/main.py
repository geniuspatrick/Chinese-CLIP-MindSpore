import os
import json
import random
import numpy as np
import logging
from math import ceil
from pathlib import Path
from time import gmtime, strftime

import mindspore as ms
from mindspore import nn
from mindspore.amp import DynamicLossScaler, StaticLossScaler
from mindspore.communication import get_group_size, get_local_rank, get_rank, init

from cn_clip.clip.model import convert_weights, resize_pos_embed, CLIP
from cn_clip.training.train import AdamW, ClipLoss, build_trainer, CallbackForCLIP
from cn_clip.training.data import get_data
from cn_clip.training.params import parse_args
from cn_clip.training.logger import setup_logging
from cn_clip.training.scheduler import cosine_lr


def random_seed(seed=42, rank=0):
    ms.set_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


# Used by https://github.com/openai/CLIP/issues/83 but not below.
# Keeping it incase needed.
def convert_models_to_fp32(model):
    for p in model.get_parameters():
        p.set_dtype(ms.float32)


def is_master(args):
    return args.rank == 0


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    if args.distributed:
        init()
        args.local_rank = get_local_rank()
        args.world_size = get_group_size()
        args.rank = get_rank()
        ms.context.set_auto_parallel_context(
            device_num=args.world_size,
            global_rank=args.rank,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )

    device = f"{ms.get_context('device_target')}:{ms.get_context('device_id')}"
    args.device = device
    return device


def check_args(args):
    """some setting are not supported yet, but will be done in the future"""
    assert args.report_training_batch_acc is False
    assert args.use_bn_sync is False, "Sync BN is not supported"
    assert args.grad_checkpointing is False, "Gradient checkpointing is not supported."
    assert args.use_flash_attention is False, "Flash attention is not supported."
    assert args.accum_freq == 1, "Accumulation is not supported."
    assert args.distillation is False


def main():
    args = parse_args()
    check_args(args)

    # Set distributed group
    ms.set_context(mode=ms.GRAPH_MODE)
    device = init_distributed_device(args)
    random_seed(args.seed)

    # Set output path
    time_suffix = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    args.log_path = os.path.join(args.logs, args.name, "out_{}.log".format(time_suffix))

    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    if is_master(args):
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)    

    assert args.precision in ['amp', 'fp16', 'fp32']

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level, args.rank)

    # Build the CLIP model
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
    model_info['use_flash_attention'] = args.use_flash_attention

    model = CLIP(**model_info)
    sd = ms.load_checkpoint(args.pretrained_weights_path)
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, sd, strict_load=True)
    if param_not_load:
        print(f"{param_not_load} in network is not loaded!")
    if ckpt_not_load:
        print(f"{ckpt_not_load} in checkpoint is not loaded!")

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp" or args.precision == "fp32":
        convert_models_to_fp32(model)
    if args.precision == "fp16":
        convert_weights(model)

    if args.freeze_vision:
        for k, v in model.visual.named_parameters():
            v.requires_grad = False
        # freeze bn running mean and variance
        if args.vision_model in ['RN50']:
            for m in model.visual.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        logging.info("The visual encoder is freezed during training.")

    # Initialize dataset and dataloader
    data = get_data(args, epoch_id=0, max_txt_length=args.context_length)

    # Initialize optimizer and lr scheduler
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n  # noqa: E731
    include = lambda n, p: not exclude(n, p)  # noqa: E731

    named_parameters = list(model.parameters_and_names())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    if args.train_data is None:
        optimizer = None
        scheduler = None
    else:
        num_batches = data["train"].dataloader.num_batches
        if args.max_steps is not None:
            args.max_epochs = ceil(args.max_steps * args.accum_freq / num_batches)
        else:
            assert args.max_epochs is not None and args.max_epochs > 0
            args.max_steps = (num_batches // args.accum_freq) * args.max_epochs
        total_steps = args.max_steps
        scheduler = cosine_lr(args.lr, args.warmup, total_steps)
        optimizer = AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
                {"order_params": model.trainable_params()},
            ],
            lr=scheduler,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

    scaler = StaticLossScaler(1024) if args.precision == "amp" else None

    # Log and save hyper-params.
    if is_master(args):
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params_{}.txt".format(time_suffix))
        with open(params_file, "w", encoding="utf-8") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                f.write(f"{name}: {val}\n")
                logging.info(f"  {name}: {val}")
        logging.info(f"Use device {args.device} for training")

    # Note for mask_ratio
    if is_master(args) and args.mask_ratio > 0 and args.vision_model in ['RN50']:
        logging.info(
            "Note: mask_ratio > 0 (FLIP strategy) is currently only implemented for VisualTransformer. "
            "It will not function for ResNet backbone."
        )

    # Optionally resume from a checkpoint
    start_epoch = 0
    steps = 0
    # Automatically restore latest checkpoint if exists
    if args.resume is None:
        latest_path = os.path.join(args.checkpoint_path, f"epoch_latest.ckpt")
        if os.path.isfile(latest_path):
            args.resume = latest_path
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logging.info(
                f"=> begin to load checkpoint '{args.resume}'"
            )
            # Restore the model weight, map model to be loaded to specified single gpu.
            # loc = "cuda:{}".format(args.local_rank)
            checkpoint = ms.load_checkpoint(args.resume)
            sd = {k: v for k, v in checkpoint["state_dict"].items() if "bert.pooler" not in k}
            # Resize the positional embedding by interpolation, if needed
            resize_pos_embed(sd, model, prefix="module.")
            # Load the state dict
            ms.load_param_into_net(model, sd, strict_load=True)
            # Restore the epoch and steps info, reload the dataset and dataloader for the resume epoch
            if not args.reset_data_offset:
                start_epoch = checkpoint["epoch"]
                steps = checkpoint["step"]
                data = get_data(args, 
                                epoch_id=start_epoch, 
                                max_txt_length=args.context_length)
            # Restore the optim state
            if not args.reset_optimizer and optimizer is not None:
                ms.load_param_into_net(optimizer, checkpoint["optimizer"], strict_load=True)
                logging.info("=> optimizer state is restored from the checkpoint")
            logging.info(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']} @ {steps} steps)"
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # determine if this worker should save logs and checkpoints.
    # only do so if it is the 0th worker.
    args.should_save = (args.logs is not None and args.logs != '' and args.logs.lower() != 'none') and is_master(args)

    loss = ClipLoss(args)
    trainer = build_trainer(
        model, loss, optimizer, amp_level=args.amp_opt_level, scaler=scaler, grad_clip_norm=args.grad_clip_norm
    )
    callbacks = [CallbackForCLIP(args, data, trainer, start_epoch)]
    trainer.train(args.max_epochs - start_epoch, data["train"].dataloader, callbacks=callbacks, dataset_sink_mode=False)


if __name__ == "__main__":
    main()
