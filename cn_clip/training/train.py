import os
import time
import math
import json
import logging
import datetime
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, context, nn, ops
from mindspore.amp import DynamicLossScaler, LossScaler, StaticLossScaler, all_finite
from mindspore.train import Callback, Model, save_checkpoint
from mindspore.train.amp import _OutputTo16, _OutputTo32, custom_mixed_precision, get_black_list, get_white_list

_logger = logging.getLogger(__name__)


def is_master(args):
    return args.rank == 0


class AllGather(nn.Cell):
    def __init__(self):
        super().__init__()
        self.all_gather = ops.AllGather()

    def construct(self, x):
        return self.all_gather(x)


class ClipLoss(nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.aggregate = args.aggregate
        self.gather_with_grad = args.gather_with_grad
        self.rank = args.rank
        self.world_size = args.world_size
        if self.world_size > 1:
            self.all_gather = AllGather()
        else:
            self.all_gather = None

    def construct(self, image_features, text_features, logit_scale):
        if self.aggregate and self.world_size > 1:
            # We gather tensors from all gpus to get more negatives to contrast with.
            if self.gather_with_grad:
                all_image_features = self.all_gather(image_features)  # w/ grad
                all_text_features = self.all_gather(text_features)  # w/ grad
            else:
                d = image_features.shape[1]
                gathered_image_features = self.all_gather(image_features).reshape(self.world_size, -1, d)  # w/o grad
                gathered_text_features = self.all_gather(text_features).reshape(self.world_size, -1, d)  # w/o grad
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[self.rank] = image_features
                gathered_text_features[self.rank] = text_features
                all_image_features = ops.reshape(gathered_image_features, (-1, d))
                all_text_features = ops.reshape(gathered_text_features, (-1, d))

            # this is needed to send gradients back everywhere.
            logits_per_image = logit_scale * all_image_features @ all_text_features.t()
            logits_per_text = logits_per_image.t()

        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

        num_logits = logits_per_image.shape[0]
        ground_truth = ops.arange(num_logits, dtype=ms.int64)

        total_loss = (
            ops.cross_entropy(logits_per_image, ground_truth) + ops.cross_entropy(logits_per_text, ground_truth)
        ) / 2
        return total_loss


class TrainStep(nn.Cell):
    """Training step with loss scale.

    The steps of model optimization are performed in the following order:
        1. calculate grad
        2. allreduce grad
        3. clip grad [optional]
        4. call optimizer
    """

    def __init__(
        self,
        network: nn.Cell,
        criterion: nn.Cell,
        optimizer: nn.Optimizer,
        scaler: LossScaler,
        grad_clip_norm: float = None,
    ):
        super().__init__()
        self.network = network.set_grad()
        self.criterion = criterion.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.scaler = scaler
        if isinstance(self.scaler, StaticLossScaler):
            self.drop_overflow = False
        elif isinstance(self.scaler, DynamicLossScaler):
            self.drop_overflow = True
        else:
            raise NotImplementedError(f"Unsupported scaler: {type(self.scaler)}")
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode == context.ParallelMode.STAND_ALONE:
            self.grad_reducer = ops.identity
        elif self.parallel_mode in (context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL):
            self.grad_reducer = nn.DistributedGradReducer(self.weights)
        else:
            raise NotImplementedError(f"When creating reducer, Got Unsupported parallel mode: {self.parallel_mode}")
        if isinstance(network, nn.Cell) and network.jit_config_dict:
            self._jit_config_dict = network.jit_config_dict

        self.clip_grad = grad_clip_norm is not None
        self.clip_value = grad_clip_norm
        self.logit_scale = None
        for n, p in self.network.parameters_and_names():
            # TODO: _OutputTo16/32 will add disgusting prefix '_backbone' on param name, unwrap it before saving!
            if n == "logit_scale" or n == "_backbone.logit_scale":
                self.logit_scale = p
        assert self.logit_scale is not None, "Cannot fetch parameter `logit_scale` from network."

        def forward_fn(image, text):
            image_features, text_features, logit_scale = network(image, text)
            loss = criterion(image_features, text_features, logit_scale)
            loss = scaler.scale(loss)
            return loss, image_features, text_features, logit_scale

        self.grad_fn = ops.value_and_grad(forward_fn, grad_position=None, weights=self.weights, has_aux=True)

    def update(self, loss, grads):
        if self.clip_grad:
            loss = ops.depend(loss, self.optimizer(ops.clip_by_global_norm(grads, clip_norm=self.clip_value)))
        else:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss

    def construct(self, *inputs):
        (loss, image_features, text_features, logit_scale), grads = self.grad_fn(*inputs)
        grads = self.grad_reducer(grads)
        loss = self.scaler.unscale(loss)
        grads = self.scaler.unscale(grads)

        if self.drop_overflow:
            status = all_finite(grads)
            if status:
                loss = self.update(loss, grads)
            loss = ops.depend(loss, self.scaler.adjust(status))
        else:
            loss = self.update(loss, grads)

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        loss = ops.depend(loss, ops.assign(self.logit_scale, ops.clamp(self.logit_scale, 0, 4.6052)))

        # if you want to get anything about training status, return it from here and logging it outside!
        return loss, image_features, text_features, logit_scale


def auto_mixed_precision(network, amp_level):
    if amp_level == "O0":
        network.to_float(ms.float32)
    elif amp_level == "O1":
        white_list = get_white_list()
        network = custom_mixed_precision(network, white_list=white_list)
    elif amp_level == "O2":
        black_list = get_black_list()
        black_list += [
            nn.GroupNorm,
            nn.SyncBatchNorm,
            nn.Softmax,
            nn.LogSoftmax,
            nn.LogSigmoid,
            nn.CrossEntropyLoss,
            nn.SoftmaxCrossEntropyWithLogits,
        ]
        network = custom_mixed_precision(network, black_list=black_list)
    elif amp_level == "O3":
        network.to_float(ms.float16)
        network = _OutputTo32(network)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))

    return network


def build_trainer(
    network: nn.Cell,
    criterion: nn.Cell,
    optimizer: nn.Cell,
    amp_level: str = "O0",
    scaler: LossScaler = None,
    grad_clip_norm: float = None,
):
    """Build Trainer.

    Args:
        network: The backbone network to train, evaluate or predict.
        criterion: The function of calculating loss.
        optimizer: The optimizer for training.
        amp_level: The level of auto mixing precision training.
            'O0': single precision for all, 'O1': half precision for white list & single precision for others
            'O2': single precision for black list & half precision for others, 'O3': half precision for all.
        scaler: The manager helps perform the steps of gradient scaling conveniently.
        grad_clip_norm: The value at which to clip gradients. Disable if it's None.

    Returns:
        mindspore.Model

    """
    network = auto_mixed_precision(network, amp_level=amp_level)
    criterion = criterion.to_float(ms.float32)
    train_step_cell = TrainStep(
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        grad_clip_norm=grad_clip_norm,
    ).set_train()
    trainer = Model(train_step_cell)
    return trainer


class CallbackForCLIP(Callback):
    def __init__(self, args, data, trainer, start_epoch=0):
        self.args = args
        self.data = data
        self.trainer = trainer
        self.start_epoch = start_epoch
        # initialize the following members to make linter happy
        self.step_ts = -1.0
        self.epoch_ts = -1.0
        self.num_batches_per_epoch = -1
        self.num_samples_per_epoch = -1
        self.sample_digits = -1

    def _get_network_from_cbp(self, cb_params):
        network = cb_params.train_network if cb_params.mode == "train" else cb_params.eval_network
        if cb_params.dataset_sink_mode:  # train_network is connected to DatasetHelper when data_sink is enable.
            return network.network  # throw an error at the beginning of 1st epoch for helper is not connected. WTF!!!
        else:
            return network

    def _get_optimizer_from_cbp(self, cb_params):
        optimizer = cb_params.optimizer
        if optimizer is None:
            network = cb_params.train_network if cb_params.mode == "train" else cb_params.eval_network
            if cb_params.dataset_sink_mode:
                optimizer = network.network.optimizer
            else:
                optimizer = network.optimizer
        if optimizer is None or not isinstance(optimizer, nn.Optimizer):
            _logger.warning(f"Failed to get valid optimizer from callback, got {type(optimizer)}")
            optimizer = None
        return optimizer

    def _get_lr_from_cbp(self, cb_params):
        optimizer = self._get_optimizer_from_cbp(cb_params)
        if optimizer.global_step < 1:
            _logger.warning(
                "`global_step` of optimizer is less than 1. It seems to be a overflow at the first step. "
                "If you keep seeing this message, it means that the optimizer never actually called."
            )
            optim_step = Tensor((0,), ms.int32)
        else:  # if the optimizer is successfully called, the global_step will actually be the value of next step.
            optim_step = optimizer.global_step - 1
        if optimizer.dynamic_lr:
            if isinstance(optimizer.learning_rate, nn.CellList):
                # return the learning rates of the first parameter if dynamic_lr
                lr = optimizer.learning_rate[0](optim_step)[0]
            else:
                lr = optimizer.learning_rate(optim_step)[0]
        else:
            lr = optimizer.learning_rate
        return lr

    def on_train_epoch_begin(self, run_context):
        self.epoch_ts = time.time()
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num + self.start_epoch
        if is_master(self.args):
            _logger.info(f"Start epoch {cur_epoch - 1}")
        dataloader = self.data["train"].dataloader
        self.num_batches_per_epoch = dataloader.num_batches // self.args.accum_freq
        self.num_samples_per_epoch = dataloader.num_samples
        self.sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        num_epochs = cb_params.epoch_num
        num_batches = cb_params.batch_num
        cur_step = cb_params.cur_step_num + self.start_epoch * num_batches
        cur_epoch = cb_params.cur_epoch_num + self.start_epoch
        epoch, completed_epoch = cur_epoch - 1, cur_epoch
        train_time = time.time() - self.epoch_ts

        val_time = 0
        if self.args.val_data is not None and self.args.valid_epoch_interval is not None and (completed_epoch % self.args.valid_epoch_interval) == 0:
            assert "val" in self.data, "Error: Valid dataset has not been built."
            val_time = time.time()
            model = self._get_network_from_cbp(cb_params).network  # TrainStep -> network(backbone, amp-ed)
            if hasattr(model, "_backbone"):  # _OutputTo32 will add a disgusting prefix '_backbone'
                model = model._backbone  # TrainStep -> network(backbone, amp-ed, last _outputTo32 unwrapped)
            evaluate(model, self.data, epoch, self.args, cur_step)
            val_time = time.time() - val_time

        # Saving checkpoints.
        if self.args.should_save:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "step": cur_step,
                "name": self.args.name,
            }
            if completed_epoch == self.args.max_epochs or (
                self.args.save_epoch_frequency > 0 and (completed_epoch % self.args.save_epoch_frequency) == 0
            ):
                t1 = time.time()
                save_path = os.path.join(self.args.checkpoint_path, f"epoch{completed_epoch}.pt")
                save_checkpoint(  # TrainStep with network(backbone), criterion, optimizer, scaler, ema, accum_grad
                    self._get_network_from_cbp(cb_params),
                    save_path,
                    append_dict=checkpoint_dict,
                )
                logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, completed_epoch, cur_step, time.time() - t1))
            # Save the latest params
            t1 = time.time()
            save_path = os.path.join(self.args.checkpoint_path, f"epoch_latest.pt")
            save_checkpoint(self._get_network_from_cbp(cb_params), save_path, append_dict=checkpoint_dict)
            logging.info("Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, completed_epoch, cur_step, time.time() - t1))

        if is_master(self.args):
            total_time = int(time.time() - self.epoch_ts)
            _logger.info(
                f"Total time since last epoch: {datetime.timedelta(seconds=total_time)}"
                f"(train: {train_time:.6f}s, val: {val_time:.6f}s), "
                f"ETA: {datetime.timedelta(seconds=(num_epochs - cur_epoch) * total_time)}"
            )

    def on_train_step_begin(self, run_context):
        self.step_ts = time.time()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        num_epochs = cb_params.epoch_num
        num_batches = cb_params.batch_num
        num_steps = num_batches * num_epochs
        # cur_x start from 1, end at num_xs, range: [1, num_xs]
        cur_step = cb_params.cur_step_num + self.start_epoch * num_batches
        cur_epoch = cb_params.cur_epoch_num + self.start_epoch
        cur_batch = (cur_step - 1) % num_batches + 1

        i, epoch = cur_batch - 1, cur_epoch - 1
        i_accum = i // self.args.accum_freq
        step = self.num_batches_per_epoch * epoch + i_accum

        batch_time = time.time() - self.step_ts
        if cb_params.dataset_sink_mode:  # if data_sink is enable, this hook is actually invoked at end of epoch
            batch_time = batch_time / self.num_batches_per_epoch
        batch_count = i_accum + 1

        if is_master(self.args) and (
            (step + 1) % self.args.log_interval == 0 or batch_count == self.num_batches_per_epoch
        ):
            batch_size = self.args.batch_size * self.args.accum_freq
            num_samples = batch_count * batch_size * self.args.world_size
            percent_complete = 100.0 * batch_count / self.num_batches_per_epoch
            outputs = cb_params.net_outputs  # todo: outputs is hardcode here
            loss_scalar = outputs[0].numpy().item()
            logit_scale_scalar = outputs[3].numpy().item()
            lr_scalar = self._get_lr_from_cbp(cb_params).numpy().item()
            _logger.info(
                f"Global Steps: {step + 1}/{self.args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{self.num_samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {loss_scalar:.6f} | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {lr_scalar:5f} | " +
                f"logit_scale: {logit_scale_scalar:.3f} | " +
                f"Global Batch Size: {batch_size * self.args.world_size}"
            )


def evaluate(model, data, epoch, args, steps):
    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))
    model.set_train(False)
    model.phase = "eval"
    dataloader = data['val'].dataloader

    cumulative_loss = Tensor(0.0)
    cumulative_i2t_acc = Tensor(0.0)
    cumulative_t2i_acc = Tensor(0.0)
    num_elements = Tensor(0.0)
    all_image_features, all_text_features = [], []
    for i, batch in enumerate(tqdm(dataloader.create_tuple_iterator(), total=dataloader.num_batches)):
        images, texts = batch
        image_features, text_features, logit_scale = model(images, texts)
        image_features, text_features, logit_scale = (
            image_features.to(ms.float32),
            text_features.to(ms.float32),
            logit_scale.to(ms.float32),
        )
        all_image_features.append(image_features.numpy())
        all_text_features.append(text_features.numpy())
        logit_scale = logit_scale.mean()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        batch_size = len(images)
        ground_truth = ops.arange(batch_size, dtype=ms.int64)
        total_loss = (
            ops.cross_entropy(logits_per_image, ground_truth) + ops.cross_entropy(logits_per_text, ground_truth)
        ) / 2
        cumulative_loss += total_loss * batch_size
        num_elements += batch_size

        cumulative_i2t_acc += ((logits_per_image.argmax(-1) == ground_truth).sum()).float()
        cumulative_t2i_acc += (logits_per_text.argmax(-1) == ground_truth).sum().float()

    loss = cumulative_loss / num_elements
    i2t_acc = cumulative_i2t_acc / num_elements
    t2i_acc = cumulative_t2i_acc / num_elements

    assert num_elements.item() == dataloader.num_samples  # sanity check

    _logger.info(
        f"Validation Result (epoch {epoch + 1} @ {steps} steps) | "
        f"Valid Loss: {loss.numpy().item():.6f} | "
        f"Image2Text Acc: {i2t_acc.numpy().item() * 100:.2f} | " 
        f"Text2Image Acc: {t2i_acc.numpy().item() * 100:.2f} | " 
        f"logit_scale: {logit_scale.numpy().item():.3f} | "
        f"Valid Batch Size: {batch_size}"
    )
