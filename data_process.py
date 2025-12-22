import os
import json
import pandas as pd
from datasets import Dataset

def main():
    RUNTIME_SCRIPT_DIR = os.environ["RUNTIME_SCRIPT_DIR"]  
    train_output_dir = os.environ["DATA_CONVERT_TRAIN_PATH"]
    os.makedirs(train_output_dir, exist_ok=True)
    validation_output_dir = os.environ["DATA_CONVERT_VALID_PATH"]
    os.makedirs(validation_output_dir, exist_ok=True)


    val_file_path = "grpo_valid.jsonl"
    val_base_name = "grpo_valid.jsonl"
    val_file_name, val_file_ext = os.path.splitext(val_base_name)
    val_file_output_path = os.path.join(validation_output_dir, val_file_name + ".parquet")
    print(f"converting {val_file_path} to {val_file_output_path}")
    df = pd.read_json(val_file_path, lines=True)
    # convert to Parquet 
    df.to_parquet(val_file_output_path, engine='pyarrow', compression='snappy')
    print(f"end converting {val_file_path} to {val_file_output_path}")
    has_validation = True

    train_file_path = "grpo_train.jsonl"
    train_base_name = "grpo_train.jsonl"
    train_file_name, train_file_ext = os.path.splitext(train_base_name)
    train_file_output_path = os.path.join(train_output_dir, train_file_name + ".parquet")

    # read train
    df = pd.read_json(train_file_path, lines=True)
    if has_validation:
        df.to_parquet(train_file_output_path, engine='pyarrow', compression='snappy')
        print(f"end converting {train_file_path} to {train_file_output_path}")
        
    else:
        dataset = Dataset.from_pandas(df)
        dataset_dict = dataset.train_test_split(test_size=0.1)
        val_file_output_path = os.path.join(validation_output_dir, "test.parquet")
        dataset_dict["train"].to_parquet(train_file_output_path)
        dataset_dict["test"].to_parquet(val_file_output_path)
        print(f"end converting {train_file_path} to {train_file_output_path}")
        print(f"end converting to {val_file_output_path}")



if __name__ == "__main__":
    main()
