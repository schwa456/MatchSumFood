import os
import datetime
import pandas as pd
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from typing import Optional, Tuple, OrderedDict
from transformers import AutoTokenizer

from src.model.model import MatchSum
from src.model.metric import MarginRankingLoss
from src.utils.lr_scheduler import get_transformer_scheduler
from src.score.rouge_score import RougeScorer


class MatchSum_Engine(pl.LightningModule):
    def __init__(self, 
        model: MatchSum,
        tokenizer_name: str,
        lr: float = None,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        num_warmup_steps: int = None,
        margin: float = 0.1,
        
        betas: Tuple[float] = (0.9, 0.999),
        save_result: bool = False,
        
        **kwargs        
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        
        self.loss_fn = MarginRankingLoss(margin=self.hparams.margin)
        self.scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer_name)
        
        self.save_hyperparameters(ignore=['model'])
        
        self.prepare_training()

    def forward(self, batch):
        """
        모든 계산을 self.model에 위임
        """
        return self.model(
            doc_ids=batch['doc_input_ids'],
            doc_mask=batch['doc_attention_mask'],
            cand_ids=batch['cand_input_ids'],
            cand_mask=batch['cand_attention_mask'],
            summary_ids=batch.get('summary_input_ids'),
            summary_mask=batch.get('summary_attention_mask')
        )

    def prepare_training(self):
        self.model.train()
        
        #if self.hparams.model_checkpoint: # Loading Model Checkpoint
        #    checkpoint = torch.load(self.model_checkpoint)
        #    assert isinstance(checkpoint, OrderedDict), 'Please Load Lightning-Format Checkpoints'
        #    assert next(iter(checkpoint)).split('.')[0] != 'model', 'This is Only For Loading The Model Checkpoints'
        #    self.model.load_state_dict(checkpoint)
        
        #if self.hparams.freeze_base: # Base Model의 파라미터는 학습 제외
        #    for p in self.model.base_model.parameters():
        #        p.requires_grad = False

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight'] # weight decay 없이 학습
        
        optim_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        ]
        optimizer = AdamW(optim_params, self.hparams.lr, betas=self.hparams.betas, eps=self.hparams.adam_epsilon)
        scheduler = get_transformer_scheduler(optimizer, self.hparams.num_warmup_steps)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}
        }
        
    def training_step(self, batch, batch_idx):
        """
        훈련 스텝이 매우 간결해집니다.
        """
        # 💡 3. 모델을 호출하여 모든 출력을 한 번에 얻습니다.
        outputs = self.forward(batch)
        
        # 💡 4. 모델이 계산한 최종 손실 값을 가져옵니다.
        loss = outputs['loss']
        
        # 로그 기록
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def on_validation_epoch_start(self):
        self._val_outputs = []
        
    def validation_step(self, batch, batch_idx):
        # 훈련 스텝과 동일하게 모델을 호출합니다.
        outputs = self.forward(batch)
        loss = outputs['loss']
        
        # 예측된 후보의 인덱스와 실제 정답 인덱스
        pred_indices = outputs['prediction']
        label_indices = batch['label']
        
        step_output = {
            'loss': loss.detach(),
            'pred_indices': pred_indices.detach(),
            'label_indices': label_indices.detach(),
            'cand_input_ids': batch['cand_input_ids'].detach(),
            'summary_input_ids': batch['summary_input_ids'].detach(),
        }
        
        self._val_outputs.append(step_output)
        
        self.log("val_loss_step", loss, prog_bar=True, logger=True, sync_dist=True)
        return step_output

    def on_validation_epoch_end(self):
        if not self._val_outputs:
            print("Validation Outputs are empty.")
            return
        
        losses = [x['loss'] for x in self._val_outputs]
        total_correct = 0
        total_samples = 0
        r1, r2, rL = [], [], []
        
        print('Calculating ROUGE Score & ACC...')
        
        for output in self._val_outputs:
            batch_size = output['pred_indices'].size(0)
            total_samples += batch_size
            total_correct += torch.sum(output['pred_indices'] == output['label_indices']).item()
            
            # --- ROUGE 점수 계산 ---
            for i in range(batch_size):
                pred_idx = output['pred_indices'][i].item()
                
                # 예측 요약문 디코딩
                pred_cand_ids = output['cand_input_ids'][i][pred_idx]
                predicted_summary = self.tokenizer.decode(pred_cand_ids, skip_special_tokens=True)
                
                # 정답 요약문 디코딩
                ref_summary_ids = output['summary_input_ids'][i]
                reference_summary = self.tokenizer.decode(ref_summary_ids, skip_special_tokens=True)
                
                # ROUGE 점수 계산
                scores = self.scorer.score(prediction=predicted_summary, target=reference_summary)
                r1.append(scores['rouge1'].fmeasure)
                r2.append(scores['rouge2'].fmeasure)
                rL.append(scores['rougeL'].fmeasure)
        
        # --- 최종 메트릭 계산 ---
        avg_loss = torch.stack(losses).mean()
        accuracy = (total_correct / total_samples) * 100
        avg_r1 = (sum(r1) / len(r1)) * 100 if r1 else 0.0
        avg_r2 = (sum(r2) / len(r2)) * 100 if r2 else 0.0
        avg_rL = (sum(rL) / len(rL)) * 100 if rL else 0.0
        
        # --- 최종 결과 logging ---
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', accuracy, prog_bar=True, sync_dist=True)
        self.log('val_rouge1', avg_r1, prog_bar=True, sync_dist=True)
        self.log('val_rouge2', avg_r2, prog_bar=True, sync_dist=True)
        self.log('val_rougeL', avg_rL, prog_bar=True, sync_dist=True)
        
        self._val_outputs.clear()
    
    def on_test_epoch_start(self):
        self._test_outputs = []


    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)


    def on_test_epoch_end(self):
        result = {
            'text': [],
            'reference summary': [],
            'candidate summary': [],
            'reference indices': [],
            'candidate indices': []
        }
        r1, r2, rL, accs = [], [], [], []

        print('calculating ROUGE score & ACC...')
        for output in self._test_outputs:
            texts = output['texts']
            ref_sums = output['ref_sums']
            can_sums = output['can_sums']

            result['reference indices'].append(output['ref_idx'])
            result['candidate indices'].append(output['can_idx'])

            for i, (ref_sum, can_sum) in enumerate(zip(ref_sums, can_sums)):
                rouge = self.scorer.score(ref_sum, can_sum)
                r1.append(rouge['rouge1'].fmeasure)
                r2.append(rouge['rouge2'].fmeasure)
                rL.append(rouge['rougeL'].fmeasure)

                if self.save_result:
                    result['text'].append(texts[i])
                    result['reference summary'].append(ref_sum)
                    result['candidate summary'].append(can_sum)

            accs.extend(output['accs'])

        r1 = 100 * (sum(r1) / len(r1))
        r2 = 100 * (sum(r2) / len(r2))
        rL = 100 * (sum(rL) / len(rL))
        acc = 100 * (sum(accs) / len(accs))

        print('rouge1: ', r1)
        print('rouge2: ', r2)
        print('rougeL: ', rL)
        print('accuracy: ', acc)

        if self.save_result:
            path = './result/{}'.format(datetime.datetime.now().strftime('%y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

            result_pd = pd.DataFrame(result)
            result_pd.to_csv(path + '/{}.csv'.format(datetime.datetime.now().strftime('%H-%M-%S')), index=False)

        self._test_outputs = [] # reset
        
    """

    def on_predict_epoch_start(self):
        self._predict_outputs = []


    def predict_step(self, batch, batch_idx):
        outputs = self.model(
            batch['encodings'],
            batch['cls_token_ids'],
        )
        preds = outputs['prediction']

        ids, texts, can_sums, can_idx = [], [], [], []

        for i, id in enumerate(batch['id']):
            ids.append(id)
            sample = self.inference_df[self.inference_df['id'] == id].squeeze()
            text = sample['text']
            texts.append('\n'.join(text))

            can_sum = get_candidate_sum(text, preds[i], self.pred_sum_size)
            can_sums.append('\n'.join(can_sum))

            pred_indices = set(preds[i][:self.pred_sum_size])
            can_idx.append(pred_indices)

        output = {
            'ids': ids,
            'texts': texts,
            'can_sums': can_sums,
            'can_idx': can_idx,
        }
        self._predict_outputs.append(output)

        return output


    def on_predict_epoch_end(self):
        result = {
            'ids': [],
            'text': [],
            'candidate summary': [],
            'candidate indices': []
        }

        for output in self._predict_outputs:
            ids = output['ids']
            texts = output['texts']
            can_sums = output['can_sums']
            can_idx = output['can_idx']

            result['ids'].extend(ids)
            result['text'].extend(texts)
            result['candidate summary'].extend(can_sums)
            result['candidate indices'].extend(can_idx)

        if self.save_result:
            path = './result/{}'.format(datetime.datetime.now().strftime('%y-%m-%d'))
            if not os.path.exists(path):
                os.makedirs(path)

            result_pd = pd.DataFrame(result)
            result_pd.to_csv(path + '/{}.csv'.format(datetime.datetime.now().strftime('%H-%M-%S')), index=False)

        self._predict_outputs = [] # reset
    """