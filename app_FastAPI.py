import numpy as np
import uvicorn
from fastapi import FastAPI
import torch
from torch.utils.data import DataLoader
from RUL_BiLSTM_CMAPSS import LSTM_RUL_Estimator, CMAPSSDataset
from pickle import load
from sklearn.metrics import mean_squared_error

# Create the app object
app = FastAPI()

# Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello World!'}

# Route with a single parameter, returns the parameter within a message
# Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/predict/{dataset}/{mode}')
def predict(dataset: str, mode: str):
    batch_size = 1024
    sequence_length = 40
    cmapss_dataset = {x: CMAPSSDataset(csv_file='data/CMAPSS/' + x + '_'+dataset+'.csv',
                                       sep=' ', seq_len=sequence_length)
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(cmapss_dataset[x], batch_size=batch_size,
                                 num_workers=0, pin_memory=True, shuffle=True)
                   for x in ['train', 'val', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = 'models/LSTM_v0_'+dataset+'.pth'
    model = LSTM_RUL_Estimator(n_features=24, hidden_dim=100, dropout=0.5,
                               seq_length=40, num_layers=3, output_dim=1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    target_scaler = load(open('data/CMAPSS/target_scaler_'+dataset+'.pkl', 'rb'))

    RMSE = []
    with torch.no_grad():
        for test_data, test_labels in dataloaders[mode]:
            test_labels = target_scaler.inverse_transform(test_labels)
            test_data = test_data.to(device)
            pred = model(test_data)
            pred = target_scaler.inverse_transform(pred.cpu())
            RMSE.append(mean_squared_error(test_labels, pred, squared=False))
        print(f'Test RMSE: {np.mean(RMSE)}')

    return {'message': f'{mode.upper()} RMSE on dataset {dataset} is: {np.mean(RMSE)}'}


# Run the API with uvicorn
# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)