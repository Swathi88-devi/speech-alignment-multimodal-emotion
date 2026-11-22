"""
PyTorch Dataset and DataLoader Module for TESS
==============================================
Loads the preprocessed TESS cache and returns batch DataLoaders for
speech, text, and multimodal fusion pipelines.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(BASE_DIR, "tess_preprocessed.pt")
TEST_CACHE = os.path.join(BASE_DIR, "tess_test_cache.pt")

class TESSDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "mel": item["mel"],             # [128, 301]
            "tokens": item["tokens"],       # [8]
            "label": torch.tensor(item["label"], dtype=torch.long),
            "emotion": item["emotion"],
            "speaker": item["speaker"],
            "word": item["word"],
            "filename": item["filename"]
        }

def get_dataloaders(batch_size=32, cache_path=DEFAULT_CACHE):
    # Fallback to test_cache if full preprocessed cache is not present
    if not os.path.exists(cache_path):
        if os.path.exists(TEST_CACHE):
            print(f"[Notice] Full training cache ({cache_path}) not found. Loading included test evaluation cache ({TEST_CACHE}).")
            cache_path = TEST_CACHE
        else:
            raise FileNotFoundError(
                f"Cache file {cache_path} not found!\n"
                f"To train from scratch, please download TESS and run python project/data/preprocess.py\n"
                f"Dataset source: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess"
            )
        
    data = torch.load(cache_path, weights_only=False)
    
    train_ds = TESSDataset(data.get("train", []))
    val_ds = TESSDataset(data.get("val", []))
    test_ds = TESSDataset(data.get("test", []))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) if len(train_ds) > 0 else None
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if len(test_ds) > 0 else None
    
    metadata = {
        "vocab": data["vocab"],
        "emotions": data["emotions"],
        "emotion_to_idx": data["emotion_to_idx"],
        "idx_to_emotion": data["idx_to_emotion"],
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds)
    }
    
    return train_loader, val_loader, test_loader, metadata

if __name__ == "__main__":
    tl, vl, te_l, meta = get_dataloaders(32)
    loader = tl if tl is not None else te_l
    sample = next(iter(loader))
    print(f"DataLoaders loaded successfully!")
    print(f"Batch mel shape: {sample['mel'].shape}")
    print(f"Batch tokens shape: {sample['tokens'].shape}")
    print(f"Batch labels shape: {sample['label'].shape}")
    print(f"Emotions: {meta['emotions']}")
