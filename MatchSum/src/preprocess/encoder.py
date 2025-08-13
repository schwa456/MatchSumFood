import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer

class BertExtEncoder(nn.Module):
    """
        문서 내 문장들의 CLS 임베딩을 인코딩하여 요약 문장인지 판단하는 스코어를 출력하는 모델
    """

    __doc__ = r"""
        cls_attention_mask prevents padding tokens from being included in softmax values 
        inside the encoder's self-attention layer.
        (cls_attention_mask: CLS 토큰이 padding 위치에 있을 경우, self-attention에서 softmax 값에 포함되지 않도록 방지)
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        intermediate_size: int,
        num_attention_head: int,
        dropout_prob: float,
    ):
        super().__init__()
        
        # save hyperparameters
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_head = num_attention_head
        self.dropout_prob = dropout_prob
        
        # define positional embedding
        self.position_embedding = PositionEmbedding(dropout_prob, hidden_size)
        # save defined BERT encoder layer 
        self.layers = nn.ModuleList([self.bert_layer() for _ in range(self.num_layers)])
        
        # last output layer of model
        # predict score(logit) whether it is extractive summary sentence for each sentence(CLS embedding)
        self.last_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.Linear(hidden_size, 1, bias=True)
        )
    
    def bert_layer(self):
        # generates one BERT encoder layer
        config = BertConfig()
        config.hidden_size = self.hidden_size
        config.intermediate_size = self.intermediate_size
        config.num_attention_heads = self.num_attention_head
        config.attention_probs_dropout_prob = self.dropout_prob
        config.hidden_dropout_prob = self.dropout_prob
        
        return BertLayer(config)
    
    def cls_attention_mask(self, cls_token_mask):
        # change cls_token_mask for using attention
        # shape: (batch, num_heads, seq_len, seq_len)
        
        attention_mask = cls_token_mask[:, None, None, :] # (batch, 1, 1, seq_len)
        attention_mask = attention_mask.permute(0, 1, 3, 2).expand(
            -1, self.num_attention_head, attention_mask.size(-1), 1) # (batch, num_head, seq_len, seq_len)
        attention_mask = (1.0 - attention_mask) * -1e18 # 마스킹된 위치는 매우 작은 값으로 설정(softmax에서 제외됨)
        
        return attention_mask
    
    def forward(self, token_embeds, cls_token_ids):
        """
        Args:
            token_embeds: Tensor of shape (B, L, H) – 전체 토큰 임베딩
            cls_token_ids: Tensor of shape (B, S) – 각 문장에 대한 CLS 위치 인덱스

        Returns:
            dict with keys:
                - 'logits': (B, S)
                - 'cls_embeddings': (B, S, H)
                - 'cls_token_mask': (B, S)
        """
        hidden_size = token_embeds.size(-1)
        
        if cls_token_ids.dim() != 2:
            raise ValueError(f'cls_token_ids must be 2D (B, S), got shape {cls_token_ids.shape}')
        
        # ✅ 문장별 임베딩 추출
        cls_token_ids_exp = cls_token_ids.unsqueeze(-1).expand(-1, -1, hidden_size)  # (B, S, H)
        cls_vec = token_embeds.gather(dim=1, index=cls_token_ids_exp)  # (B*N, S, H)

        # ✅ 마스킹: padding된 문장은 무시
        cls_mask = (cls_token_ids != -1).float()  # (B*N, S)
        #cls_mask = torch.Tensor(1.0)

        # ✅ logits 계산
        logits = self.last_layer(cls_vec).squeeze(-1)  # (B*N, S)

        return {
            'logits': logits,                 # (B, S)
            'cls_embeddings': cls_vec,        # (B, S, H)
            'cls_token_mask': cls_mask        # (B, S)
        }


class PositionEmbedding(nn.Module):
    
    def __init__(
        self,
        dropout_prob: float,
        dim: int,
        max_len: int = 5000
    ):
    
        # sin  - cos 기반 위치 임베딩 생성
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *- (math.log(10000.0) / dim)))
        
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        
        pe = pe.unsqueeze(0) # batch 차원 추가 (1, max_len, dim)
        
        super().__init__()
        self.register_buffer('pe', pe) # 학습되지 않도록 buffer로 등록
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dim = dim
    
    def get_embed(self, embed):
        return self.pe[:, :embed.size(1)]
    
    def forward(self, embed, step=None):
        embed = embed * math.sqrt(self.dim) # 임베딩 스케일링
        # step이 주어지면 해당 위치의 임베딩을 더하고, 아니면 전체 위치 임베딩을 더함
        # step은 현재 위치의 index(예: 배치 내 문장 위치)
        if step:
            embed = embed + self.pe[:, step][:, None, :] # 특정 위치만 더할 경우
        else:
            embed = embed + self.pe[: :embed.size(1)] # 전체 위치를 더할 경우
            
        embed = self.dropout(embed)
        
        return embed
        