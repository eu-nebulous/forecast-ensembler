"""Script for torch lighning module for ensembling"""

"""Script for deep learning networks """
import os

import numpy as np
import pandas as pd
# PyTorch Lightning
import pytorch_lightning as pl
import torch
import torch.nn as nn
from dl_dataset import DatasetHistMask
from dl_models import BaseCNNNetworkMask, EnsemblerRegressorModel
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader


class Ensembler(pl.LightningModule):
    def __init__(
        self, df, models_list=["0", "1", "2", "3", "4", "5"], lr=0.0002, series_len=11
    ):
        super().__init__()
        self.df = df
        self.models_list = models_list
        self.series_len = series_len
        self.prepare()
        self.lr = lr
        # self.model = EnsemblerRegressorModel(
        #     forecasters_num=len(models_list),
        #     n_inputs=[len(models_list) * 3 + 5, 32, 64],
        #     n_outputs=[32, 64, 128],
        #     series_len=self.series_len,
        # )
        self.model = BaseCNNNetworkMask(num_feat=len(models_list) * 3 + 5)

        self.loss = nn.L1Loss()
        self.val_loss = []
        self.val_mae = []
        self.train_loss_1 = []
        self.train_mae = []
        self.val_predictions = {}
        self.val_true = []
        self.val_pred = []

    def smape(self, y_true, y_pred):
        return (
            1
            / len(y_true)
            * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        )

    def mape(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / (y_true))) * 1

    def prepare(self):
        self.train_set = DatasetHistMask(
            self.df,
            models_list=self.models_list,
            start_idx=0,
            last_idx=4000,
            series_len=self.series_len,
        )

        self.val_set = DatasetHistMask(
            self.df,
            models_list=self.models_list,
            start_idx=4000,
            last_idx=None,
            series_len=self.series_len,
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=64, num_workers=4)

    def forward(self, x):
        y_pred = self.model(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        output = self.model((batch[0], batch[1]))
        loss = self.loss(batch[2], output)
        self.train_loss_1.append(loss.item())
        self.train_mae.append(torch.mean(torch.abs(batch[2] - output)).detach())
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model((batch[0], batch[1]))
        loss = self.loss(batch[2], output)
        self.val_loss.append(loss)
        self.val_mae.append(torch.mean(torch.abs(batch[2] - output)))
        self.val_true.extend(batch[-1].squeeze().tolist())
        self.val_pred.extend(output.squeeze().tolist())
        metric = {"val_loss": loss}
        self.log_dict(metric)
        return loss

    def validation_epoch_end(self, val_step_outputs):
        print("val epoch end")
        print("MAE", np.mean(np.abs(np.array(self.val_true) - np.array(self.val_pred))))
        self.val_true = []
        self.val_pred = []

    def training_epoch_end(self, val_step_outputs):
        print(np.mean(np.array(self.train_loss_1)), " train loss")
        self.train_loss = []
        self.train_mae = []

    def predict(self):
        pass

    def configure_optimizers(self):
        optimizers = [torch.optim.Adam(self.parameters(), lr=self.lr)]
        schedulers = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[0], mode="min", factor=0.2, patience=4, verbose=True
                ),
                "monitor": "val_loss",  # Default: val_loss
                "interval": "epoch",
                "frequency": 1,
            },
        ]
        return optimizers, schedulers


PREDICTIONS_PATH = "cpu_predictions"
Y_COLUMN = "real"
P_COLUMN = "prediction"
STEP = 1
PRED_LEN = 5


def create_data():
    data = pd.DataFrame()
    for i, file in enumerate(os.listdir(PREDICTIONS_PATH)):
        tmp_data = pd.read_csv(os.path.join(PREDICTIONS_PATH, file))
        tmp_data["index"] = range(tmp_data.shape[0])
        tmp_data = tmp_data[tmp_data.index % PRED_LEN == 0]
        data[f"{i}_value"] = tmp_data[P_COLUMN]
        data["y"] = tmp_data[Y_COLUMN]

    data.index = range(data.shape[0])
    return data


data = create_data()
data["series_id"] = 0
models_list = ["0", "1", "2", "3", "4", "5"]
data = create_data()
data["series_id"] = 0
models_list = ["0", "1", "2", "3", "4", "5"]
data.head(3)

network = Ensembler(data, models_list=models_list)

trainer = pl.Trainer(max_epochs=50, gpus=0, auto_lr_find=False, gradient_clip_val=0.15)
trainer.fit(network)
