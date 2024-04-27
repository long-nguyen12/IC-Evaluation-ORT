import torch
from torch.utils import data

from data_utils.utils import preprocess_caption
from utils.instance import Instance

import json
import os
import numpy as np
import cv2 as cv
from typing import Dict, List, Any

class FeatureDataset(data.Dataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__()
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = vocab

        # quesion-answer pairs
        self.annotations = self.load_json(json_data)

        # image features
        self.image_features_path = config.FEATURE_PATH.FEATURES

    def load_json(self, json_data: Dict) -> List[Dict]:
        annotations = []
        for ann in json_data["annotations"]:
            # find the appropriate image
            for image in json_data["images"]:
                if image["id"] == ann["image_id"]:
                    annotation = {
                        "caption": preprocess_caption(ann["caption"], self.vocab.tokenizer),
                        "image_id": ann["image_id"],
                        "filename": image["file_name"]
                    }
                    break

            annotations.append(annotation)

        return annotations

    # def load_features(self, image_id: int) -> Dict[str, Any]:
    #     feature_file = os.path.join(self.image_features_path, f"{image_id}.npz")
    #     features = np.load(feature_file, allow_pickle=True)
    #     # features = {key: value for key, value in features.items() if key != 'num_bbox'}

    #     return features
    def load_feature(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npz")
        feature = np.load(feature_file, allow_pickle=True)["features"].copy()

        return feature

    def load_boxes(self, image_id: int) -> np.ndarray:
        
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npz")
        features = np.load(feature_file, allow_pickle=True)
        height, width = features['image_h'], features['image_w']
        
        boxes = features['bbox']
        boxes = boxes/np.array([width, height, width,height])
        boxes = np.clip(boxes, 0.0, 1.0, dtype=np.float32)
        return boxes
    @property
    def captions(self):
        return [ann["caption"] for ann in self.annotations]

    def __getitem__(self, idx: int):
        item = self.annotations[idx]
        caption = self.vocab.encode_caption(item["caption"])
        
        shifted_right_caption = torch.zeros_like(caption).fill_(self.vocab.padding_idx)
        shifted_right_caption[:-1] = caption[1:]
        caption = torch.where(caption == self.vocab.eos_idx, self.vocab.padding_idx, caption) # remove eos_token in caption
        
        # features = self.load_features(self.annotations[idx]["image_id"])
        
        visual = self.load_feature(self.annotations[idx]["image_id"])
        boxes = self.load_boxes(self.annotations[idx]["image_id"])
        return Instance(caption_tokens=caption, shifted_right_caption_tokens=shifted_right_caption, visual=visual, boxes=boxes)
        # return Instance(
        #     caption_tokens=caption,
        #     shifted_right_caption_tokens=shifted_right_caption,
        #     **features,
        # )
    def __len__(self) -> int:
        return len(self.annotations)

class DictionaryDataset(data.Dataset):
    def __init__(self, json_path: str, vocab, config) -> None:
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        # vocab
        self.vocab = vocab

        # captions
        self.image_ids, self.filenames, self.captions_with_image = self.load_json(json_data)

        # images
        self.image_features_path = config.FEATURE_PATH.FEATURES

    # def load_features(self, image_id: int) -> Dict[str, Any]:
    #     feature_file = os.path.join(self.image_features_path, f"{image_id}.npz")
    #     features = np.load(feature_file, allow_pickle=True)
        
    #     return features

    def load_features(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npz")
        feature = np.load(feature_file, allow_pickle=True)["features"].copy()

        return feature

    def load_boxes(self, image_id: int) -> np.ndarray:
        feature_file = os.path.join(self.image_features_path, f"{image_id}.npz")
        boxes = np.load(feature_file, allow_pickle=True)["bbox"].copy()

        return boxes
    
    def __len__(self):
        return len(self.image_ids)

    def load_json(self, json_data: Dict) -> List[Dict]:
        examples = {}
        filenames = {}
        for image in json_data["images"]:
            examples[image["id"]] = []
            filenames[image["id"]] = image["file_name"]

        for ann in json_data["annotations"]:
            caption = preprocess_caption(ann["caption"], self.vocab.tokenizer)
            caption = " ".join(caption)
            examples[ann["image_id"]].append(caption)

        image_ids = []
        captions_with_image = []
        for image_id, captions in examples.items():
            image_ids.append(image_id)
            captions_with_image.append(captions)

        return image_ids, list(filenames.values()), captions_with_image

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        filename = self.filenames[idx]
        # features = self.load_features(image_id)
        captions = self.captions_with_image[idx]

        # return Instance(
        #     filename=filename,
        #     captions=captions,
        #     **features
        # )

        visual = self.load_features(image_id)
        boxes = self.load_boxes(image_id)
        return Instance(filename=filename, caption=captions, visual=visual, boxes=boxes)
        

class ImageDataset(DictionaryDataset):
    # This class is designed especially for visualizing purposes
    def __init__(self, json_path: str, vocab, config) -> None:
        super().__init__(json_path, vocab, config)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        filename = self.filenames[idx]
        image_file = os.path.join(self.image_path, filename)
        image = cv.imread(image_file)
        image = cv.resize(image, (512, 512), interpolation=cv.INTER_AREA)

        features = self.load_features(image_id)
        captions = self.captions_with_image[idx]

        return Instance(
            **features,
            captions=captions
        )
