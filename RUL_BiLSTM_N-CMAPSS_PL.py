import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pickle import load
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def plot_loss_history(train_loss, val_loss):
    plt.figure(figsize=(20, 8))
    plt.plot(train_loss.index.tolist(), train_loss.tolist(),
             lw=3, label='Train Loss')
    plt.plot(val_loss.index.tolist(), val_loss.tolist(),
             lw=3, label='Validation Loss')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()


class CMAPSSDataset(Dataset):
    """N-CMAPSS dataset."""

    def __init__(self, csv_file, sep=' ', seq_len=40):
        """
        :param csv_file (string): Path to the csv dataset file.
        """
        self.df_cmapss = pd.read_csv(csv_file, sep=sep)
        self.df_data = self.df_cmapss.loc[:, 'unit':'phi']
        # drop 'unit' and column 0
        self.feature_columns = self.df_data.columns[1:]
        self.targets = self.df_cmapss[['unit', 'RUL']]
        self.seq_len = seq_len

        self.seq_gen = (list(self.gen_sequence(self.df_data[self.df_data['unit'] == id],
                                               self.feature_columns))
                        for id in self.df_data['unit'].unique() if
                        len(self.df_data[self.df_data['unit'] == id]) >= seq_len)

        self.seq_data = np.concatenate(list(self.seq_gen)).astype(np.float32)

        self.targets_gen = [self.gen_targets(self.targets[self.targets['unit'] == id], ['RUL'])
                            for id in self.targets['unit'].unique() if
                            len(self.targets[self.targets['unit'] == id]) >= seq_len]

        self.seq_targets = np.concatenate(self.targets_gen).astype(np.float32)

    # Function to generate sequences of shape: (samples, time steps, features)
    def gen_sequence(self, id_df, feature_columns):
        """ Only consider sequences that meets the window-length, no padding is used. This means for testing
        we need to drop those which are below the window-length. An alternative would be to pad sequences so that
        we can use shorter ones """
        data_array = id_df[feature_columns].values
        num_elements = data_array.shape[0]
        if (num_elements != self.seq_len):
            for start, stop in zip(range(0, num_elements - self.seq_len), range(self.seq_len, num_elements)):
                yield data_array[start:stop, :]
        else:
            yield data_array[:num_elements, :]

    # Function to generate labels
    def gen_targets(self, id_df, label):
        data_array = id_df[label].values
        num_elements = data_array.shape[0]
        return data_array[self.seq_len:num_elements, :]

    def __len__(self):
        return len(self.seq_data) - (self.seq_len - 1)

    def __getitem__(self, idx):
        data = self.seq_data[idx]
        target = self.seq_targets[idx]

        data = torch.tensor(data)
        target = torch.tensor(target)

        return data, target


class CMAPSSDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, seq_len=1,
                 batch_size=1024, num_workers=0):
        super().__init__()
        self.train_data = train_data
        self.train_dataset = None
        self.val_data = val_data
        self.val_dataset = None
        self.test_data = test_data
        self.test_dataset = None
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

    # def setup(self, stage=None):
    #     if stage in (None, "fit"):
    #         self.train_dataset = CMAPSSDataset(csv_file=self.train_data, sep=' ',
    #                                            seq_len=self.seq_len)
    #         self.val_dataset = CMAPSSDataset(csv_file=self.val_data, sep=' ',
    #                                          seq_len=self.seq_len)
    #
    #     if stage in (None, "test"):
    #         self.test_dataset = CMAPSSDataset(csv_file=self.test_data, sep=' ',
    #                                           seq_len=self.seq_len)

    def setup(self, stage=None):
        self.train_dataset = CMAPSSDataset(csv_file=self.train_data, sep=' ',
                                           seq_len=self.seq_len)
        self.val_dataset = CMAPSSDataset(csv_file=self.val_data, sep=' ',
                                         seq_len=self.seq_len)
        self.test_dataset = CMAPSSDataset(csv_file=self.test_data, sep=' ',
                                          seq_len=self.seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class LSTMRul(pl.LightningModule):
    def __init__(self, n_features, hidden_dim=50, dropout=0.2, seq_len=40, num_layers=2,
                 output_dim=1, criterion=None, learning_rate=1e-3):
        super(LSTMRul, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate

        # Define the LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.linear = nn.Linear(in_features=hidden_dim * 2, out_features=output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        pred = torch.relu(self.linear(lstm_out))
        return pred[:, -1, :]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_loss', loss)
        return loss


if __name__ == '__main__':
    batch_size = 8192
    sequence_length = 40
    EPOCHS = 20

    data_module = CMAPSSDataModule(train_data='data/N-CMAPSS/train_DS03.csv',
                                   val_data='data/N-CMAPSS/val_DS03.csv',
                                   test_data='data/N-CMAPSS/test_DS03.csv',
                                   seq_len=sequence_length,
                                   batch_size=batch_size,
                                   num_workers=30)
    data_module.setup()

    # check sample data from the training set
    # sample_data, sample_labels = next(iter(data_module.train_dataloader()))

    model_params = dict(
        n_features=32,
        hidden_dim=100,
        seq_len=sequence_length,
        num_layers=6,
        dropout=0.5,
        output_dim=1,
        criterion=torch.nn.MSELoss(),
        learning_rate=1e-3,
    )
    model = LSTMRul(**model_params)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total number of parameters')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} parameters to train')

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="LSTM-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        mode="min"
    )

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10)

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=EPOCHS,
        gpus=8
        # progress_bar_refresh_rate=30
        # check_val_every_n_epoch=2
    )

    trainer.fit(model, data_module)

    # load the scaler
    target_scaler = load(open('data/N-CMAPSS/target_scaler_DS03.pkl', 'rb'))

    model.eval()
    RMSE = []
    with torch.no_grad():
        for test_data, test_labels in data_module.test_dataloader():
            test_labels = target_scaler.inverse_transform(test_labels)
            pred = model(test_data)
            pred = target_scaler.inverse_transform(pred.cpu())
            RMSE.append(mean_squared_error(test_labels, pred, squared=False))
        print(f'Test RMSE: {np.mean(RMSE)}')
