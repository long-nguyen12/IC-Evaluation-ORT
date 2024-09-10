import torch
from torch import nn
from torch.nn import functional as F

from models.modules.attentions import MultiHeadAttention
from models.utils import generate_padding_mask, generate_sequential_mask, sinusoid_encoding_table
from models.modules.positionwise_feed_forward import PositionWiseFeedForward
from models.modules.containers import Module, ModuleList
from builders.decoder_builder import META_DECODER
from builders.text_embedding_builder import build_text_embedding
from builders.pretrained_language_model_builder import build_pretrained_language_model
from models.utils import clones, box_relational_embedding
from models.modules.pos_embeddings import SinusoidPositionalEmbedding
import numpy as np

class DecoderLayer(Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config.SELF_ATTENTION)
        self.enc_attn = MultiHeadAttention(config.ENC_ATTENTION)
        
        self.pwff = PositionWiseFeedForward(config.ENC_ATTENTION)

    def forward(self, queries, keys, values, self_padding_mask, self_attention_mask, enc_attention_mask, **kwargs):
        self_att = self.self_attn(queries, queries, queries, padding_mask=self_padding_mask, attention_mask=self_attention_mask, **kwargs)        
        enc_att = self.enc_attn(self_att, keys, values, padding_mask=self_padding_mask, attention_mask=enc_attention_mask, **kwargs)
        
        ff = self.pwff(enc_att)
        ff = ff.masked_fill(self_padding_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=0)
        
        return ff

class MeshedDecoderLayer(Module):
    def __init__(self, config):
        super(MeshedDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config.SELF_ATTENTION)
        self.enc_attn = MultiHeadAttention(config.ENC_ATTENTION)
        self.pwff = PositionWiseFeedForward(config.ENC_ATTENTION)

        self.fc_alphas = nn.ModuleList([
            nn.Linear(2*config.D_MODEL, config.D_MODEL) for _ in range(config.N_ENCODER_LAYERS)
        ])

        self.nlayers = config.N_ENCODER_LAYERS

        self.init_weights()

    def init_weights(self):
        for ith, _ in enumerate(self.fc_alphas):
            nn.init.xavier_uniform_(self.fc_alphas[ith].weight)
            nn.init.constant_(self.fc_alphas[ith].bias, 0)

    def forward(self, queries, keys, values, self_padding_mask, self_attention_mask, enc_attention_mask, **kwargs):
        self_att = self.self_attn(queries, queries, queries, padding_mask=self_padding_mask, attention_mask=self_attention_mask, **kwargs)

        enc_atts = []
        for ith in range(self.nlayers):
            enc_att = self.enc_attn(self_att, keys[:, ith], values[:, ith], padding_mask=self_padding_mask, attention_mask=enc_attention_mask, **kwargs)
            enc_atts.append(enc_att)

        alphas = []
        for ith, (fc_alpha, enc_att) in enumerate(zip(self.fc_alphas, enc_atts)):
            alpha = torch.sigmoid(fc_alpha(torch.cat([self_att, enc_att], dim=-1)))
            alphas.append(alpha)

        out_att = 0
        for alpha, enc_att in zip(alphas, enc_atts):
            out_att += alpha*enc_att
        out_att /= (self.nlayers)**0.5


        ff = self.pwff(out_att)
        ff = ff.masked_fill(self_padding_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=0)
        
        return ff

@META_DECODER.register()
class Decoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, config, vocab):
        super(Decoder, self).__init__()
        
        self.d_model = config.D_MODEL
        self.max_len = vocab.max_caption_length
        self.padding_idx = vocab.padding_idx
        self.N = config.LAYERS

        self.word_emb = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len=self.max_len+1,
                                                                            d_model=config.D_MODEL, padding_idx=0), freeze=True)
        self.layers = ModuleList([DecoderLayer(config.ATTENTION) for _ in range(config.LAYERS)])
        self.fc = nn.Linear(config.D_MODEL, len(vocab), bias=False)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, caption_tokens, encoder_features, encoder_attention_mask):
        b_s, seq_len = caption_tokens.shape[:2]
        caption_padding_masks = generate_padding_mask(caption_tokens, self.padding_idx).to(caption_tokens.device)
        caption_self_attention_masks = generate_sequential_mask(seq_len).to(caption_tokens.device)
        caption_self_attention_masks = torch.logical_or(caption_padding_masks, caption_self_attention_masks)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, caption_self_attention_masks], -1)
            caption_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(caption_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(caption_padding_masks.squeeze(1).squeeze(1), 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        embedded_captions, _ = self.word_emb(caption_tokens)
        out = embedded_captions + self.pos_emb(seq)
        for layer in self.layers:
            out = layer(queries=out, 
                        keys=encoder_features,
                        values=encoder_features,
                        self_padding_mask=caption_padding_masks,
                        self_attention_mask=caption_self_attention_masks,
                        enc_attention_mask=encoder_attention_mask)

        out = self.fc(out)

        return F.log_softmax(out, dim=-1)

@META_DECODER.register()
class MeshedDecoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, config, vocab):
        super().__init__()
        
        self.d_model = config.D_MODEL
        self.max_len = vocab.max_caption_length
        self.padding_idx = vocab.padding_idx
        self.N = config.LAYERS

        self.word_emb = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len=self.max_len+1,
                                                                            d_model=config.D_MODEL, padding_idx=0), freeze=True)
        self.layers = ModuleList([MeshedDecoderLayer(config.ATTENTION) for _ in range(config.LAYERS)])
        self.fc = nn.Linear(config.D_MODEL, len(vocab), bias=False)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, caption_tokens, encoder_features, encoder_attention_mask):
        b_s, seq_len = caption_tokens.shape[:2]
        caption_padding_masks = generate_padding_mask(caption_tokens, self.padding_idx).to(caption_tokens.device)
        caption_self_attention_masks = generate_sequential_mask(seq_len).to(caption_tokens.device)
        caption_self_attention_masks = torch.logical_or(caption_padding_masks, caption_self_attention_masks)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, caption_self_attention_masks], -1)
            caption_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(caption_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(caption_padding_masks.squeeze(1).squeeze(1), 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        embedded_captions, _ = self.word_emb(caption_tokens)
        out = embedded_captions + self.pos_emb(seq)
        for layer in self.layers:
            out = layer(queries=out, 
                        keys=encoder_features,
                        values=encoder_features,
                        self_padding_mask=caption_padding_masks,
                        self_attention_mask=caption_self_attention_masks,
                        enc_attention_mask=encoder_attention_mask)

        out = self.fc(out)

        return F.log_softmax(out, dim=-1)

@META_DECODER.register()
class AdaptiveDecoder(Module):
    def __init__(self, config, vocab):
        super(AdaptiveDecoder, self).__init__()

        self.d_model = config.D_MODEL
        self.max_len = vocab.max_caption_length
        self.padding_idx = vocab.padding_idx
        self.N = config.LAYERS

        self.word_emb = build_text_embedding(config.TEXT_EMBEDDING)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len=self.max_len+1,
                                                                            d_model=config.D_MODEL, padding_idx=0), freeze=True)
        self.layers = ModuleList([DecoderLayer(config.ATTENTION) if i < config.LAYERS else DecoderLayer(config.ADAPTIVE_ATTENTION) 
                                                for i in range(config.LAYERS + 1)])
        self.fc = nn.Linear(config.D_MODEL, len(vocab), bias=False)

        # load and froze the language model
        self.language_model = build_pretrained_language_model(config.LANGUAGE_MODEL)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, caption_tokens, encoder_features, encoder_attention_mask):
        b_s, seq_len = caption_tokens.shape[:2]
        caption_padding_masks = generate_padding_mask(caption_tokens, self.padding_idx).to(caption_tokens.device)
        caption_self_attention_masks = generate_sequential_mask(seq_len).to(caption_tokens.device)
        caption_self_attention_masks = torch.logical_or(caption_padding_masks, caption_self_attention_masks)
        
        b_s, seq_len = caption_tokens.shape[:2]
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, caption_self_attention_masks], -1)
            caption_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(caption_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(caption_padding_masks, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        # get the language_signals
        _, language_signals = self.language_model(caption_tokens)

        out = self.word_emb(caption_tokens) + self.pos_emb(seq)
        for layer in self.layers:
            out = layer(queries=out, 
                        keys=encoder_features,
                        values=encoder_features,
                        language_signals=language_signals,
                        self_padding_mask=caption_padding_masks,
                        self_attention_mask=caption_self_attention_masks,
                        enc_attention_mask=encoder_attention_mask)

        out = self.fc(out)

        return F.log_softmax(out, dim=-1)

class IntegratedDecoderLayer(Module):
    def __init__(self, config):
        super(IntegratedDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(config.SELF_ATTENTION)
        self.region_attn = MultiHeadAttention(config.SELF_ATTENTION)
        self.feature_attn = MultiHeadAttention(config.SELF_ATTENTION)
        
        self.pwff = PositionWiseFeedForward(config.ENC_ATTENTION)

    def forward(self, queries, keys, values, self_padding_mask, self_attention_mask, enc_attention_mask, **kwargs):
        self_att = self.self_attn(queries, queries, queries, padding_mask=self_padding_mask, attention_mask=self_attention_mask, **kwargs)            
        region_att = self.region_attn(self_att, keys, keys, padding_mask=self_padding_mask, attention_mask=enc_attention_mask, **kwargs)
        feature_att = self.region_attn(self_att, values, values, padding_mask=self_padding_mask, attention_mask=enc_attention_mask, **kwargs)

        enc_att = (region_att + feature_att) / np.sqrt(2)

        ff = self.pwff(enc_att)
        ff = ff.masked_fill(self_padding_mask.squeeze(1).squeeze(1).unsqueeze(-1), value=0)
        
        return ff

@META_DECODER.register()
class IntegratedDecoder(Module):
    "Generic N layer decoder with masking."
    def __init__(self, config, vocab):
        super(IntegratedDecoder, self).__init__()
        
        self.d_model = config.D_MODEL
        self.max_len = vocab.max_caption_length
        self.padding_idx = vocab.padding_idx
        self.N = config.LAYERS

        self.pos_embedding = SinusoidPositionalEmbedding(config.D_MODEL)
        self.word_emb = build_text_embedding(config.TEXT_EMBEDDING, vocab)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len=self.max_len+1,
                                                                            d_model=config.D_MODEL, padding_idx=0), freeze=True)
        self.layers = ModuleList([IntegratedDecoderLayer(config.ATTENTION) for _ in range(config.LAYERS)])
        self.fc = nn.Linear(config.D_MODEL, len(vocab), bias=False)
        
        self.d_g = config.D_MODEL // 8
        self.fc_gs = clones(nn.Linear(self.d_g, 1), 8)
        self.layer_norm = nn.LayerNorm(config.D_MODEL)

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).bool())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, caption_tokens, features, boxes, padding_mask):
        
        # begin: cross attention preparation
        relative_geometry_embeddings = box_relational_embedding(boxes, dim_g=self.d_g, trignometric_embedding=True)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, self.d_g)
        bs, nk, _, _ = relative_geometry_embeddings.shape
        box_size_per_head = [bs, 1, nk, nk]
        relative_geometry_weights_per_head = [fc_g(flatten_relative_geometry_embeddings).view(box_size_per_head) for fc_g in self.fc_gs]
        relative_geometry_weights = torch.cat(relative_geometry_weights_per_head, dim=1) # (bs, h, nk, nk)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        
        out_feat = self.layer_norm(features) + self.pos_embedding(features)
        e_b_s, nq = out_feat.shape[:2]
        
        out_feat_v = out_feat.view(e_b_s, nk, 8, 64).permute(0, 2, 1, 3)
        relative_geometry_weights = torch.matmul(relative_geometry_weights, out_feat_v).permute(0, 2, 1, 3).contiguous().view(e_b_s, nq, self.d_model)
        # end: cross attentaiton preparation
        
        b_s, seq_len = caption_tokens.shape[:2]
        caption_padding_masks = generate_padding_mask(caption_tokens, self.padding_idx).to(caption_tokens.device)
        caption_self_attention_masks = generate_sequential_mask(seq_len).to(caption_tokens.device)
        caption_self_attention_masks = torch.logical_or(caption_padding_masks, caption_self_attention_masks)
        
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, caption_self_attention_masks], -1)
            caption_self_attention_masks = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(caption_tokens.device)  # (b_s, seq_len)
        seq = seq.masked_fill(caption_padding_masks.squeeze(1).squeeze(1), 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        embedded_captions, _ = self.word_emb(caption_tokens)
        out = embedded_captions + self.pos_emb(seq)
        for layer in self.layers:
            out = layer(queries=out, 
                        keys=relative_geometry_weights,
                        values=out_feat,
                        self_padding_mask=caption_padding_masks,
                        self_attention_mask=caption_self_attention_masks,
                        enc_attention_mask=padding_mask)

        out = self.fc(out)

        return F.log_softmax(out, dim=-1)