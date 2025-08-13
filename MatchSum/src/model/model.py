import torch
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict
from .metric import *

class MatchSum(nn.Module):

    __doc__ = r"""
        Implementation of the paper;
        https://arxiv.org/abs/2004.08795
    """
    
    def __init__(
            self,
            candidate_num: Optional[int],
            tokenizer: Optional[str],
            margin: Optional[float],
            hidden_size: int = 768
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.candidate_num = candidate_num

        self.encoder = AutoModel.from_pretrained(tokenizer)

        self.loss_fn = MarginRankingLoss(margin=margin)

            # Check whether Model Params are frozen
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print("[Frozen]", name)
                
    def get_embedding(self, input_ids, attention_mask):
        """[CLS] 토큰의 임베딩을 추출하는 헬퍼 함수"""
        # 입력 차원이 (batch, seq_len) 또는 (batch, num_cands, seq_len) 일 수 있음
        is_batched = input_ids.dim() == 2
        
        if not is_batched:
            batch_size, num_cands, seq_len = input_ids.size()
            input_ids = input_ids.view(-1, seq_len)
            attention_mask = attention_mask.view(-1, seq_len)
            
        # BERT 인코더를 통과
        outputs = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        
        # [CLS] 토큰 임베딩 추출
        embedding = outputs.last_hidden_state[:, 0, :]
        
        if not is_batched:
            embedding = embedding.view(batch_size, num_cands, self.hidden_size)
        
        return embedding


    def forward(self, 
                doc_ids: torch.Tensor,
                doc_mask: torch.Tensor,
                cand_ids: torch.Tensor,
                cand_mask: torch.Tensor,
                summary_ids: Optional[torch.Tensor],
                summary_mask: Optional[torch.Tensor]):
        """
        모델의 순전파 및 손실 계산을 모두 수행
        """
        # 공유된 인코더를 통해 각 입력의 임베딩을 계산
        doc_emb = self.get_embedding(doc_ids, doc_mask) # (batch, hidden_size)
        cand_emb = self.get_embedding(cand_ids, cand_mask) # (batch, num_cands, hidden_size)
        
        # 문서와 후보 요약문 간의 코사인 유사도를 점수로 계산
        # doc_emb를 (batch, 1, hidden_size)로 확장하여 broadcasting
        doc_emb_expanded = doc_emb.unsqueeze(1).expand_as(cand_emb)
        cand_score = nn.functional.cosine_similarity(cand_emb, doc_emb_expanded, dim=-1) # (batch, num_cands)
        
        # 추론 시에는 점수와 인덱스만 반환
        if summary_ids is None:
            pred_idx = torch.argmax(cand_score, dim=1)
            return {
                'score': cand_score,
                'prediction': pred_idx,
            }
        
        # 훈련 시에는 gold summary와의 점수 및 최종 손실까지 계산
        sum_emb = self.get_embedding(summary_ids, summary_mask) # (batch, hidden_size)
        summary_score = nn.functional.cosine_similarity(sum_emb, doc_emb, dim=-1) # (batch)
        
        # 손실함수 호출
        loss = self.loss_fn(score=cand_score, summary_score=summary_score)
        
        pred_idx = torch.argmax(cand_score, dim=1)
        
        return {
            'score': cand_score,
            'summary_score': summary_score,
            'prediction': pred_idx,
            'loss': loss
        }