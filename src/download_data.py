"""
Download fraud detection dataset from Kaggle
"""
import os
import sys
from pathlib import Path


def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_json.exists():
        print("\n" + "="*60)
        print("Kaggle API credentials not found!")
        print("="*60)
        print("\nTo download datasets from Kaggle, you need to:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. This will download kaggle.json")
        print("4. Place kaggle.json in: %USERPROFILE%\\.kaggle\\kaggle.json (Windows)")
        print("   or ~/.kaggle/kaggle.json (Linux/Mac)")
        print("\nAlternatively, you can manually download the dataset:")
        print("1. Visit: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Download creditcard.csv")
        print("3. Place it in the 'data/' directory")
        print("="*60 + "\n")
        return False

    return True


def download_dataset(dataset_name: str, output_dir: str):
    """Download dataset from Kaggle"""
    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        print(f"Downloading dataset: {dataset_name}")
        print(f"Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)

        # Download and unzip
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True)

        print(f"\nDataset downloaded successfully to {output_dir}")
        return True

    except Exception as e:
        print(f"\nError downloading dataset: {str(e)}")
        return False


def main():
    """Main function to download datasets"""
    print("="*60)
    print("Kaggle Dataset Downloader")
    print("="*60 + "\n")

    # Check credentials
    if not setup_kaggle_credentials():
        sys.exit(1)

    # Get project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"

    # Available datasets
    datasets = {
        "1": {
            "name": "mlg-ulb/creditcardfraud",
            "description": "Credit Card Fraud Detection (284,807 transactions)",
            "file": "creditcard.csv"
        },
        "2": {
            "name": "kartik2112/fraud-detection",
            "description": "Synthetic Financial Fraud (6M+ transactions)",
            "file": "fraudTest.csv"
        },
        "3": {
            "name": "ealaxi/paysim1",
            "description": "PaySim Mobile Money Transactions (6M+ transactions)",
            "file": "PS_20174392719_1491204439457_log.csv"
        }
    }

    print("Available datasets:")
    for key, info in datasets.items():
        print(f"{key}. {info['description']}")
        print(f"   Dataset: {info['name']}")
        print(f"   Expected file: {info['file']}\n")

    choice = input("Select dataset (1-3) [1]: ").strip() or "1"

    if choice not in datasets:
        print("Invalid choice. Defaulting to dataset 1.")
        choice = "1"

    selected = datasets[choice]
    print(f"\nDownloading: {selected['description']}")

    success = download_dataset(selected["name"], str(data_dir))

    if success:
        print("\n" + "="*60)
        print("Download completed successfully!")
        print("="*60)
        print(f"\nYou can now run the training scripts.")
        print(f"Data location: {data_dir}")
    else:
        print("\n" + "="*60)
        print("Download failed.")
        print("="*60)
        print("\nPlease manually download the dataset from Kaggle:")
        print(f"https://www.kaggle.com/datasets/{selected['name']}")
        print(f"And place {selected['file']} in: {data_dir}")


if __name__ == "__main__":
    main()
