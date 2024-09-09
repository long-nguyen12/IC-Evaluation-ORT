import torch

from builders.encoder_builder import build_encoder
from builders.decoder_builder import build_decoder
from builders.vision_embedding_builder import build_vision_embedding
from builders.model_builder import META_ARCHITECTURE
from utils.instance import Instance
from .base_transformer import BaseTransformer


@META_ARCHITECTURE.register()
class ObjectRelationModel(BaseTransformer):
    def __init__(self, config, vocab):
        super().__init__(vocab)

        self.device = torch.device(config.DEVICE)

        self.vision_embedding = build_vision_embedding(config.VISION_EMBEDDING)
        self.decoder = build_decoder(config.DECODER, vocab)

    def forward(self, input_features):
        region_features = input_features.visual
        region_boxes = input_features.boxes

        region_features, region_padding_mask = self.vision_embedding(region_features)

        caption_tokens = input_features.caption_tokens
        output = self.decoder(
            caption_tokens=caption_tokens,
            features=region_features,
            boxes=region_boxes,
            padding_mask=region_padding_mask,
        )

        return output

    def encoder_forward(self, input_features):
        region_features = input_features.visual
        region_boxes = input_features.boxes

        region_features, region_padding_mask = self.vision_embedding(region_features)

        encoder_features = self.encoder(
            features=region_features,
            padding_mask=region_padding_mask,
            boxes=region_boxes,
        )

        return encoder_features, region_padding_mask
