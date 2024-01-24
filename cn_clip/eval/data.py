import os
import logging
import json
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import base64
from io import BytesIO
import lmdb

from mindspore.dataset import SequentialSampler, GeneratorDataset, ImageFolderDataset
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import Decode, Resize, ToTensor, Normalize, Inter

from cn_clip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class EvalTxtDataset:
    def __init__(self, jsonl_filename, max_txt_length=24):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)

        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.texts = []
        with open(jsonl_filename, "r", encoding="utf-8") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                text_id = obj['text_id']
                text = obj['text']
                self.texts.append((text_id, text))
        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')

        self.max_txt_length = max_txt_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_id, text = self.texts[idx]
        text = tokenize([_preprocess_text(str(text))], context_length=self.max_txt_length)[0]
        return text_id, text


class EvalImgDataset:
    def __init__(self, lmdb_imgs, resolution=224):
        assert os.path.isdir(lmdb_imgs), "The image LMDB directory {} not exists!".format(lmdb_imgs)

        logging.debug(f'Loading image LMDB from {lmdb_imgs}.')

        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)
        self.cursor_imgs = self.txn_imgs.cursor()
        self.iter_imgs = iter(self.cursor_imgs)
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))
        logging.info("The specified LMDB directory contains {} images.".format(self.number_images))

        self.transform = self._build_transform(resolution)

    def _build_transform(self, resolution):
        return Compose([
            Resize((resolution, resolution), interpolation=Inter.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), is_hwc=False),
        ])

    def __len__(self):
        return self.number_images

    def __getitem__(self, idx):
        img_id, image_b64 = next(self.iter_imgs)
        if img_id == b"num_images":
            img_id, image_b64 = next(self.iter_imgs)

        img_id = img_id.tobytes()
        image_b64 = image_b64.tobytes()

        img_id = int(img_id.decode(encoding="utf8", errors="ignore"))
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image = self.transform(image)[0]

        return img_id, image


@dataclass
class DataInfo:
    dataloader: GeneratorDataset
    sampler: SequentialSampler


def get_eval_txt_dataset(args, max_txt_length=24):
    input_filename = args.text_data
    dataset = EvalTxtDataset(
        input_filename,
        max_txt_length=max_txt_length)
    num_samples = len(dataset)
    sampler = SequentialSampler()

    dataloader = GeneratorDataset(
        dataset,
        column_names=["text_id", "text"],
        num_parallel_workers=1,
        sampler=sampler,
    ).batch(batch_size=args.text_batch_size, drop_remainder=False)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


def get_eval_img_dataset(args):
    lmdb_imgs = args.image_data
    dataset = EvalImgDataset(
        lmdb_imgs, resolution=fetch_resolution(args.vision_model))
    num_samples = len(dataset)
    sampler = SequentialSampler()

    dataloader = GeneratorDataset(
        dataset,
        column_names=["img_id", "image"],
        num_parallel_workers=1,
        sampler=sampler,
    ).batch(batch_size=args.img_batch_size, drop_remainder=False)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_zeroshot_dataset(args, preprocess_fn):
    dataloader = ImageFolderDataset(
        args.datapath,
        num_parallel_workers=args.num_workers,
    ).map(
        operations=[Decode(to_pil=True), preprocess_fn],
        input_columns="image",
    ).batch(args.img_batch_size)

    return DataInfo(dataloader, None)
