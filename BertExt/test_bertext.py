import hydra
from src import *

@hydra.main(version_base=None, config_path='./config', config_name='bertext_config')
def test(cfg: DictConfig):
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    
    model = BertExt(**cfg.model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_checkpoint)
    
    # Load Test Dataset
    test_df = get_test_df(cfg.dataset.path, cfg.dataset.use_df)
    test_dataset = BertExt_Dataset(test_df, tokenizer, cfg.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    engine = BertExt_Engine(model, test_df=test_df, **cfg.engine)
    cfg_trainer = Config_Trainer(cfg.trainer)()
    trainer = pl.Trainer(**cfg_trainer, logger=False)
    
    from torch.serialization import add_safe_globals
    from omegaconf import ListConfig, DictConfig, AnyNode
    from omegaconf.base import ContainerMetadata, Metadata
    from typing import Any
    from collections import defaultdict
    
    add_safe_globals([ListConfig, ContainerMetadata, Any, list, defaultdict, dict, int, DictConfig, AnyNode, Metadata])
    
    if 'test_checkpoint' in cfg:
        trainer.test(engine, test_loader, ckpt_path=cfg.test_checkpoint)
    else:
        raise RuntimeError('No Checkpoint is Given')
    
if __name__ == '__main__':
    test()