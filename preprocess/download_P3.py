import datasets

def download_and_show_splits(dataset_name):
    """
    Downloads a dataset from Hugging Face and shows all of its splits.
    
    Args:
    dataset_name (str): The name of the dataset to download.
    
    Returns:
    None
    """
    # Download the dataset
    dataset = datasets.load_dataset(dataset_name)
    
    # Display the splits
    print(f"Splits available for the dataset '{dataset_name}':")
    for split in dataset.keys():
        print(f"- {split} (Number of examples: {len(dataset[split])})")

if __name__ == "__main__":
    os.environ['HF_HOME'] = "/net/nfs.cirrascale/allennlp/xinxil/"
    download_and_show_splits("bigscience/P3")