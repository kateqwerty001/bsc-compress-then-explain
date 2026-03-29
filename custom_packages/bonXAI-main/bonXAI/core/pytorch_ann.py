import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
import copy

class BasicNeuralNetwork(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_outputs)
        )
    def forward(self, x):
        return self.network(x)
        
        
class PyTorchANN(BaseEstimator):
    """
    A wrapper for a PyTorch Artificial Neural Network that is compatible with scikit-learn.
    The network architecture is fixed: Input -> 100 -> 100 -> Output.
    """
    def __init__(self, task_type='classification', epochs=1000, lr=0.001, batch_size=512, random_state=42):
        self.task_type = task_type
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.model_ = None 
        torch.manual_seed(self.random_state)

    def fit(self, X, y, verbose=False):
        """
        Trains the neural network with early stopping.
        
        Args:
            X (pd.DataFrame): Training features.
            y (np.ndarray): Training target.
        """
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        n_inputs = X_train.shape[1]

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

        if self.task_type == 'classification':
            n_outputs = len(np.unique(y)) 
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        elif self.task_type == 'regression':
            n_outputs = 1
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
            criterion = nn.MSELoss()
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")
        

        self.model_ = BasicNeuralNetwork(n_inputs, n_outputs)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        

        patience = 10  # How many epochs to wait for improvement before stopping
        patience_counter = 0
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(self.epochs):
            self.model_.train() 
            for batch_X, batch_y in train_loader:
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- Validation Step ---
            self.model_.eval() 
            val_loss = 0.0
            with torch.no_grad():
                val_outputs = self.model_(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()

            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

            # --- Early Stopping Logic ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model_.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_model_state:
            self.model_.load_state_dict(best_model_state)
        
        return self

    def predict(self, X):
        """
        Makes predictions on new data.
        """
        if not self.model_:
            raise RuntimeError("You must call fit before calling predict.")
            
        X_tensor = torch.tensor(X, dtype=torch.float32)

        self.model_.eval() 
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            if self.task_type == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                return predicted.numpy()
            else: 
                return outputs.numpy().flatten() 

    def predict_proba(self, X):
        """
        Calculates class probabilities for new data.
        Only available for classification tasks.
        """
        if self.task_type != 'classification':
            raise AttributeError(f"predict_proba is not available for task_type='{self.task_type}'")
        
        if not self.model_:
            raise RuntimeError("You must call fit before calling predict_proba.")

        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        self.model_.eval() 
        with torch.no_grad():
            logits = self.model_(X_tensor)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.numpy()