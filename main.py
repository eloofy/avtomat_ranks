import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import MeanAbsoluteError
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class TimeSeriesLSTM(pl.LightningModule):
    """
    PyTorch Lightning module for time series prediction using LSTM.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, lr):
        """
        Initialize the TimeSeriesLSTM model.

        Parameters:
            input_size (int): Number of features in the input.
            hidden_size (int): Number of features in the hidden state of the LSTM.
            output_size (int): Number of features in the output.
            num_layers (int): Number of LSTM layers.
            lr (float): Learning rate for optimization.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lr = lr

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_size)
        self.loss = nn.L1Loss()
        self.mae = MeanAbsoluteError()

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :])  # Apply ReLU to the last LSTM output
        out = self.fc1(out)
        out = self.relu(out)
        return self.fc2(out)

    def configure_optimizers(self):
        """
        Configure optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.

        Parameters:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for PyTorch Lightning.

        Parameters:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        x, y = batch
        output = self(x)
        y = y.squeeze(dim=-1)
        loss = self.loss(output, y)
        mae = self.mae(y, output)

        self.log('mae_val', mae, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step for PyTorch Lightning.

        Parameters:
            batch (tuple): Input and target batch.
            batch_idx (int): Index of the batch.

        Returns:
            dict: Dictionary containing test loss, true values, and predicted values.
        """
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y_true)
        self.log('test_loss', loss, prog_bar=True)
        return {'loss': loss, 'y_true': y_true, 'y_pred': y_pred}

    def on_test_start(self, outputs: list):
        """
        Test epoch end for PyTorch Lightning.

        Parameters:
            outputs (list): List of dictionaries containing test loss, true values, and predicted values.
        """
        avg_loss = torch.stack([out['loss'] for out in outputs]).mean()

        y_true = torch.cat([out['y_true'] for out in outputs], dim=0).numpy()
        y_pred = torch.cat([out['y_pred'] for out in outputs], dim=0).detach().numpy()

        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='True Values', marker='o')
        plt.plot(y_pred, label='Predicted Values', marker='o')
        plt.title('True vs Predicted Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Scaled Values')
        plt.legend()
        plt.show()

    def predict(self, data):
        """
        Make predictions on new data.

        Parameters:
            data (numpy.ndarray): New data for prediction.

        Returns:
            numpy.ndarray: Predicted values.
        """
        self.eval()
        with torch.no_grad():
            data = torch.from_numpy(data).float().view(-1, self.input_size, 1)
            if torch.cuda.is_available():
                data = data.to('cuda')
            prediction = self(data).cpu().numpy()
        return prediction

    def save_model(self, path):
        """
        Save the model's state dict to a file.

        Parameters:
            path (str): File path to save the model.
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved at {path}")

    def load_model(self, path):
        """
        Load the model's state dict from a file.

        Parameters:
            path (str): File path to load the model.
        """
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

def prepare_data(data):
    """
    Prepare data for training.

    Parameters:
        data (pd.DataFrame): Input data.

    Returns:
        torch.utils.data.DataLoader: DataLoader for training.
    """
    X = torch.from_numpy(data['num_orders'].values[:-1]).float().view(-1, 1, 1)
    y = torch.from_numpy(data['num_orders'].values[1:]).float().view(-1, 1, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=64, shuffle=False)


def main():
    """
    Main function to execute time series prediction.
    """
    # Load data
    df = pd.read_csv('data_main.csv')

    # Split data
    train_size = int(len(df) * 0.80)
    train_data, val_data = df[0:train_size], df[train_size:]
    test_data = df[train_size:]

    # Prepare data
    train_dataloader = prepare_data(train_data)
    val_dataloader = prepare_data(val_data)
    test_dataloader = prepare_data(test_data)

    # Initialize and train the model
    input_size = 1
    output_size = 1
    hidden_size = 512
    num_layers = 10
    lr = 0.0005

    model = TimeSeriesLSTM(input_size, hidden_size, output_size, num_layers, lr)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, train_dataloader, val_dataloader)
    # trainer.test(dataloaders=test_dataloader)


if __name__ == "__main__":
    main()