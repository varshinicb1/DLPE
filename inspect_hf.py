from datasets import load_dataset
import pandas as pd

try:
    print("Loading dataset tanoor890/indian_crop_cleaned_train...")
    dataset = load_dataset("tanoor890/indian_crop_cleaned_train", split="train", streaming=True)
    
    print("Peeking at first 5 rows...")
    head = list(dataset.take(5))
    df = pd.DataFrame(head)
    print(df.columns)
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
