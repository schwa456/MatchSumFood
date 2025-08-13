#import argparse
import hydra
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src import *

#parser = argparse.ArgumentParser()
#parser.add_argument("--config-name", dest="config_name", default=None, type=str)
#args = parser.parse_args()

os.environ["HYDRA_FULL_ERROR"] = "1"

def custom_collate_fn(batch):
    # doc, summary, label은 이미 텐서이므로 stack으로 묶습니다.
    doc_input_ids = torch.stack([item['doc_input_ids'] for item in batch])
    doc_attention_mask = torch.stack([item['doc_attention_mask'] for item in batch])
    summary_input_ids = torch.stack([item['summary_input_ids'] for item in batch])
    summary_attention_mask = torch.stack([item['summary_attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # 후보 요약문(candidate)은 개수가 다르므로 padding이 필요합니다.
    # cand_input_ids의 리스트를 가져옵니다. 각 요소의 shape: (num_cands, seq_len)
    cand_ids_list = [item['cand_input_ids'] for item in batch]
    cand_mask_list = [item['cand_attention_mask'] for item in batch]

    # pad_sequence를 사용하여 배치 내에서 후보 개수를 통일시킵니다.
    # batch_first=True -> (batch_size, max_num_cands, seq_len)
    padded_cand_ids = pad_sequence(cand_ids_list, batch_first=True, padding_value=0)
    padded_cand_mask = pad_sequence(cand_mask_list, batch_first=True, padding_value=0)

    return {
        'doc_input_ids': doc_input_ids,
        'doc_attention_mask': doc_attention_mask,
        'cand_input_ids': padded_cand_ids,
        'cand_attention_mask': padded_cand_mask,
        'summary_input_ids': summary_input_ids,
        'summary_attention_mask': summary_attention_mask,
        'label': labels
    }

@hydra.main(version_base=None, config_path='./config', config_name='train_config')
def train(cfg: DictConfig):
    # Data Preparation
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer)
    
    train_df = pd.read_csv('./data/train_df.csv')
    val_df = pd.read_csv('./data/val_df.csv')
    
    train_data = MatchSumDataset(train_df, tokenizer, cfg.max_seq_len)
    val_data = MatchSumDataset(val_df, tokenizer, cfg.max_seq_len)
    
    train_loader = DataLoader(train_data, cfg.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_data, cfg.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    # Initializing Model and Engine
    matchsum_model = MatchSum(
        candidate_num=cfg.model.candidate_num,
        tokenizer=cfg.model.tokenizer,
        margin=cfg.model.margin
    )
    
    total_training_steps = len(train_loader) * cfg.trainer.Trainer.max_epochs
    
    engine = MatchSum_Engine(
        model=matchsum_model,
        tokenizer_name=cfg.model.tokenizer,
        lr=cfg.engine.lr,
        weight_decay=cfg.engine.weight_decay,
        adam_epsilon=cfg.engine.adam_epsilon,
        num_warmup_steps=cfg.engine.num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    # Setting logger and Trainer
    logger = My_WandbLogger(**cfg.log, save_artifact=False)
    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(
        **cfg_trainer,
        logger=logger,
        num_sanity_val_steps=0
    )
    
    trainer.fit(engine, train_loader, val_loader)
    
    wandb.finish()

if __name__ == '__main__':
    train()