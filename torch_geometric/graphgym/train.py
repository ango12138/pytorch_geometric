import os.path as osp
import warnings
from typing import Optional

from torch.utils.data import DataLoader

from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir, get_ckpt_path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule


class GraphGymDataModule(LightningDataModule):
    def __init__(self):
        self.loaders = create_loader()
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]


def train(model: GraphGymModule, datamodule, logger: bool = True,
          trainer_config: Optional[dict] = None):
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    callbacks = []
    if logger:
        callbacks.append(LoggerCallback())

    save_last = False
    if cfg.train.auto_resume:
        if cfg.train.epoch_resume < 0:
            ckpt_path = 'last'
            save_last = True
        else:
            ckpt_path = get_ckpt_path(cfg.train.epoch_resume)
            if not osp.exists(ckpt_path):
                raise ValueError(
                    f"Can't find checkpoint of epoch {cfg.train.epoch_resume}")
    else:
        ckpt_path = None

    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(
            dirpath=get_ckpt_dir(), every_n_epochs=cfg.train.ckpt_period,
            save_top_k=1 if cfg.train.ckpt_clean else -1, filename='{epoch}',
            auto_insert_metric_name=False, save_last=save_last)
        callbacks.append(ckpt_cbk)

    if cfg.accelerator == 'cuda' and cfg.devices == 1:
        strategy = pl.strategies.SingleDeviceStrategy(f'cuda:{cfg.device}')
    else:
        strategy = 'auto'

    trainer_config = trainer_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        enable_progress_bar=False,
        max_epochs=cfg.optim.max_epoch,
        check_val_every_n_epoch=cfg.train.eval_period,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=strategy,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=datamodule)
