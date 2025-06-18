"""
This program downloads, formats and saves as json the training data we use to train our tokenizers.
We use the NKJP1M corpus, with 1 million tokens, a sub-corpus of the National Corpus of Polish (Narodowy Korpus Języka Polskiego).
The data is available on HuggingFace: https://huggingface.co/datasets/ipipan/nlprepl
"""

import json
import os
from datasets import load_dataset
from typing import Dict, List, Optional, Any

def build_dataset(dataset_splits: Dict[str, Any], feature_name: str, num_examples: Optional[int] = None) -> List[str]:
    """
    Process and combine multiple dataset splits into a single list of text samples.

    Args:
        dataset_splits (Dict[str, Dataset]): Dictionary of dataset splits (e.g., 'train', 'test', 'validation').
        feature_name (str): The key used to extract text from each dataset example.
        num_examples (Optional[int]): Maximum number of examples to include.

    Returns:
        List[str]: A list of extracted and cleaned text strings.
    """
    
    clean_dataset = []

    # Iterate over all dataset splits
    for _, dataset in dataset_splits.items():
        # Iterate over each example in the current split
        for example in dataset:
            # Extract the desired feature from the example
            value = example.get(feature_name)
            if value is not None:
                clean_dataset.append(value)
                # Stop early if the number of desired examples is reached
                if num_examples is not None and len(clean_dataset) >= num_examples:
                    return clean_dataset

    return clean_dataset

def main() -> None:
    """
    Loads all splits of the dataset, combines them, and saves as a JSON file.
    """
    
    splits = ['train', 'test', 'validation']
    dataset_splits = {}

    # Load each dataset split into a dictionary
    for split in splits:
        dataset_splits[split] = load_dataset("ipipan/nlprepl", name="by_name-nkjp-conllu", split=split)

    # Combine all splits into one dataset
    combined = build_dataset(dataset_splits, feature_name='text', num_examples=5000)
    print(f"Splits combined." if combined else "No data loaded.")

    # Ensure the output directory exists
    output_path = 'data/train-1000.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save combined dataset to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(combined)} examples to {output_path}")

if __name__ == '__main__':
    main()
