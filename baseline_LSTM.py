import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def print_loss_history(train_loss, validation_loss, logscale=False):
    loss = train_loss
    val_loss = validation_loss
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, color='red', label='Training loss')
    plt.plot(epochs, val_loss, color='green', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if logscale:
        plt.yscale('log')
    plt.show()
    return


# Training Function
def train_model(model, loss_function, optimizer, num_epochs=25):
    since = time.time()
    loss_history = {'train': [], 'test': []}

    for epoch in range(1, num_epochs):
        print('\nEpoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training phase and a validation phase at every 10 epochs
        for phase in ['train', 'test']:
            # Set model to training or evaluation mode
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0

            # Iterate over data.
            for idx, (inputs, labels) in tqdm(enumerate(dataloaders[phase]),
                                              leave=True,
                                              total=len(dataloaders[phase])):

                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.float()

                if phase == 'train':
                    # Pytorch accumulates gradients, we need to clear them out before each instance.
                    model.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                elif phase == 'test':
                    with torch.no_grad():
                        outputs = model(inputs)
                        loss = loss_function(outputs, labels)

                running_loss += loss.item()

            epoch_loss = running_loss / len(cmapss_dataset[phase])
            loss_history[phase].append(epoch_loss)
            if epoch % 5 == 0:
                if phase == 'train':
                    train_stats = '{} ==> Loss:{:.4f}'.format(phase.upper(), epoch_loss)
                else:
                    # print(train_stats)
                    print(train_stats+' -- {} ==> Loss:{:.4f}'.format(phase.upper(), epoch_loss))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model, loss_history


class LSTM_RUL_Estimator(nn.Module):

    def __init__(self, n_features, hidden_dim, seq_length, num_layers=2, output_dim=1):
        super(LSTM_RUL_Estimator, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        # Define the LSTM layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=output_dim)

    def forward(self, input):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).to(device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).to(device)

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        lstm_out, (hn, cn) = self.lstm(input.float(), (h0, c0))
        # lstm_out, self.hidden = self.lstm(input.float(), self.hidden)

        # Index hidden state of last time step
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        pred = self.linear(lstm_out[:, -1, :])
        return pred


def collate_batch(batch):
    data = [item[0] for item in batch]
    data = pad_sequence(data, batch_first=True)
    targets = [item[1].unsqueeze_(0) for item in batch]
    targets = pad_sequence(targets, batch_first=True)

    return data, targets


class CMAPSSDataset(Dataset):
    """CMAPSS dataset."""

    def __init__(self, csv_file, sep=' ', sequence_length=40):
        """
        :param csv_file (string): Path to the csv dataset file.
        """
        self.df_cmapss = pd.read_csv(csv_file, sep=sep)
        self.df_data = self.df_cmapss.iloc[:, 3:27]
        self.targets = self.df_cmapss['RUL']
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        if (idx+self.sequence_length) > len(self.df_data):
            indexes = list(range(idx, len(self.df_data)))
            index_target = len(self.df_data)-1
        else:
            indexes = list(range(idx, idx + self.sequence_length))
            index_target = idx+self.sequence_length-1
        data = self.df_data.iloc[indexes, :].values
        target = self.targets.iloc[index_target]
        return torch.tensor(data), torch.tensor(target)


if __name__ == '__main__':
    batch_size = 192
    sequence_length = 40
    cmapss_dataset = {x: CMAPSSDataset(csv_file='data/CMAPSSData/'+x+'_FD001.csv',
                                       sep=' ', sequence_length=sequence_length)
                      for x in ['train', 'test']}

    dataloaders = {x: DataLoader(cmapss_dataset[x], batch_size=batch_size,
                                 num_workers=0, pin_memory=True, collate_fn=collate_batch)
                   for x in ['train', 'test']}

    # Get some random training examples
    sample_data, sample_labels = next(iter(dataloaders['train']))
    # sample_data.shape
    # sample_labels.shape

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # device='cpu'
    lstm_model = LSTM_RUL_Estimator(n_features=sample_data.shape[2],
                                    hidden_dim=100,
                                    seq_length=sequence_length,
                                    num_layers=2, output_dim=1)

    total_params = sum(p.numel() for p in lstm_model.parameters())
    print(f'{total_params:,} total number of parameters')
    total_trainable_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} parameters to train')
    print(lstm_model)

    # for idx, (data, labels) in enumerate(dataloaders['train']):
    #     print('idx: ', idx)
    #     print(data.shape)
    #     print(labels.shape)

    lstm_model = lstm_model.to(device)

    loss_function = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

    lstm_model, loss_history = train_model(lstm_model,
                                           loss_function,
                                           optimizer,
                                           num_epochs=30)

    test_data, test_labels = next(iter(dataloaders['test']))
    lstm_model.eval()
    with torch.no_grad():
        for test_data, test_labels in iter(dataloaders['test']):
            # test_data.shape
            # test_labels.shape
            test_data = test_data.to(device)
            pred = lstm_model(test_data)
            print(f'Actual: {test_labels}, Predicted: {pred}')
