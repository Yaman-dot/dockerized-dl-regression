import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class ModelTrainer():
    def __init__(self, dataset, test_size = 0.2, batch_size=32, lr=1e-4):
        #split the data
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessData(data=dataset, test_size=test_size)

        #make sure its a tensor
        self.convert_to_tensors()
        self.create_datasets()
        self.create_dataloaders(batch_size=batch_size)
        

    def create_datasets(self):
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
    def preprocess_data(self, data, test_size):
        X_train, X_test, y_train, y_test = train_test_split(self.data, test_size=self.test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    def convert_to_tensors(self):
        X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)

        y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        self.X_train_tensor = X_train_tensor
        self.X_test_tensor = X_test_tensor
        self.y_train_tensor = y_train_tensor
        self.y_test_tensor = y_test_tensor

    def create_datasets(self):
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)
    
    def create_dataloaders(self, batch_size):
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=True
        )
    def train(self, model, lr, epochs, batch_size, device):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        print(f"Starting training for {epochs} epochs \n")
        model.to(device)
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for X, y in progress_bar:
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                
                y_pred = model(X)
                
                loss = loss_fn(y_pred, y)
                loss.backward()
                
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_dataloader)

            model.eval()
            test_loss = 0
            for X_test, y_test in (self.test_dataloader):
                X_test, y_test = X_test.to(device), y_test.to(device)
                with torch.inference_mode():
                    y_pred_test = model(X_test)
                    loss = loss_fn(y_pred_test, y_test)
                    test_loss += loss.item()

            avg_test_loss = test_loss / len(self.test_dataloader)
            print(f"Epoch {epoch+1} completed. Final Loss: {loss.item():.4f}")