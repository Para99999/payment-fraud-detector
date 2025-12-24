"""
Data loading and preprocessing for payment transaction data
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


class TransactionDataset(Dataset):
    """Custom Dataset for transaction data"""

    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class PaymentDataLoader:
    """Load and preprocess payment transaction data from Kaggle"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_kaggle_data(self, dataset_name: str = "fraud") -> pd.DataFrame:
        """
        Load fraud detection dataset
        Expected dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        or synthetic financial fraud dataset
        """
        # Try multiple common fraud dataset filenames
        possible_files = [
            os.path.join(self.data_path, "creditcard.csv"),
            os.path.join(self.data_path, "PS_20174392719_1491204439457_log.csv"),
            os.path.join(self.data_path, "fraud_data.csv"),
            os.path.join(self.data_path, "transactions.csv")
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                print(f"Loading data from {file_path}")
                df = pd.read_csv(file_path)
                return df

        raise FileNotFoundError(
            f"No fraud dataset found in {self.data_path}. "
            "Please download from Kaggle and place in data/ directory"
        )

    def preprocess_creditcard_fraud(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess credit card fraud dataset (already has V1-V28 features)"""
        # Separate features and labels
        if 'Class' in df.columns:
            X = df.drop('Class', axis=1).values
            y = df['Class'].values
        else:
            X = df.values
            y = None

        # Scale features
        X = self.scaler.fit_transform(X)

        return X, y

    def preprocess_paysim_fraud(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess PaySim synthetic fraud dataset"""
        # Expected columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
        #                   nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

        # Feature engineering
        features = []

        # Numerical features
        numerical_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                         'oldbalanceDest', 'newbalanceDest']

        for col in numerical_cols:
            if col in df.columns:
                features.append(df[col].values.reshape(-1, 1))

        # Encode transaction type
        if 'type' in df.columns:
            le = LabelEncoder()
            type_encoded = le.fit_transform(df['type']).reshape(-1, 1)
            features.append(type_encoded)
            self.label_encoders['type'] = le

        # Derived features
        if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
            balance_change = (df['oldbalanceOrg'] - df['newbalanceOrig']).values.reshape(-1, 1)
            features.append(balance_change)

        # Combine all features
        X = np.hstack(features)

        # Scale features
        X = self.scaler.fit_transform(X)

        # Extract labels
        if 'isFraud' in df.columns:
            y = df['isFraud'].values
        elif 'Class' in df.columns:
            y = df['Class'].values
        else:
            y = None

        return X, y

    def auto_preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Automatically detect and preprocess dataset format"""
        # Check which dataset format this is
        if 'V1' in df.columns and 'V28' in df.columns:
            print("Detected Credit Card Fraud dataset format")
            return self.preprocess_creditcard_fraud(df)
        elif 'type' in df.columns and 'nameOrig' in df.columns:
            print("Detected PaySim fraud dataset format")
            return self.preprocess_paysim_fraud(df)
        else:
            print("Unknown format, attempting generic preprocessing")
            return self.preprocess_generic(df)

    def preprocess_generic(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generic preprocessing for unknown dataset formats"""
        # Find label column (common names)
        label_cols = ['isFraud', 'Class', 'fraud', 'is_fraud', 'label']
        label_col = None

        for col in label_cols:
            if col in df.columns:
                label_col = col
                break

        if label_col:
            y = df[label_col].values
            X = df.drop(label_col, axis=1)
        else:
            y = None
            X = df

        # Select only numerical columns
        X = X.select_dtypes(include=[np.number]).values

        # Scale features
        X = self.scaler.fit_transform(X)

        return X, y

    def create_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        batch_size: int = 64,
        random_state: int = 42
    ) -> Dict[str, DataLoader]:
        """Create train, validation, and test dataloaders"""

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state, stratify=y
        )

        # Second split: separate train and validation
        val_size = val_split / (train_split + val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )

        # Create datasets
        train_dataset = TransactionDataset(X_train, y_train)
        val_dataset = TransactionDataset(X_val, y_val)
        test_dataset = TransactionDataset(X_test, y_test)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def get_feature_dim(self, X: np.ndarray) -> int:
        """Get the feature dimension"""
        return X.shape[1]


def download_kaggle_dataset(dataset: str, output_dir: str):
    """
    Download dataset from Kaggle

    Example datasets:
    - mlg-ulb/creditcardfraud
    - kartik2112/fraud-detection
    """
    import kaggle

    print(f"Downloading {dataset} from Kaggle...")
    kaggle.api.dataset_download_files(dataset, path=output_dir, unzip=True)
    print(f"Dataset downloaded to {output_dir}")


if __name__ == "__main__":
    # Example usage
    data_loader = PaymentDataLoader("../data")

    # Download dataset (requires Kaggle API credentials)
    # download_kaggle_dataset("mlg-ulb/creditcardfraud", "../data")

    # Load and preprocess data
    df = data_loader.load_kaggle_data()
    print(f"Loaded {len(df)} transactions")

    X, y = data_loader.auto_preprocess(df)
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Fraud ratio: {y.sum() / len(y):.4f}")

    # Create dataloaders
    dataloaders = data_loader.create_dataloaders(X, y, batch_size=64)
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    print(f"Test batches: {len(dataloaders['test'])}")
