import torch
from torch import nn

from models.modules.containers import Module
from models.modules.beam_search import BeamSearch
from utils.instance import InstanceList

class BaseTransformer(Module):
    def __init__(self, vocab):
        super(BaseTransformer, self).__init__()

        self.vocab = vocab
        self.max_len = vocab.max_caption_length
        self.eos_idx = vocab.eos_idx

        self.register_state('encoder_features', None)
        self.register_state('encoder_padding_mask', None)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encoder_forward(self, input_features: InstanceList):
        raise NotImplementedError
    
    def decoder_forward(self, input_features):
        raise NotImplementedError

    def forward(self, input_features: InstanceList):
        raise NotImplementedError

    def step(self, t, prev_output):
        bs = self.region_features.shape[0]
        if t == 0:
            it = torch.zeros((bs, 1)).long().fill_(self.vocab.bos_idx).to(self.device)
        else:
            it = prev_output
        print(it.shape)
        output = self.decoder(
            caption_tokens=it,
            features=self.region_features,
            boxes=self.region_boxes,
            padding_mask=self.region_padding_mask,
        )

        return output

    def beam_search(self, input_features: InstanceList, batch_size: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
        beam_search = BeamSearch(model=self, max_len=self.max_len, eos_idx=self.eos_idx, beam_size=beam_size, 
                            b_s=batch_size, device=self.device)
        with self.statefulness(batch_size):
            self.region_features, self.region_padding_mask, self.region_boxes = self.encoder_forward(input_features)
            output = beam_search.apply(out_size, return_probs, **kwargs)

        return output