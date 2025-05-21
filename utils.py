import numpy as np
import torch
from hippie.dataloading import MultiModalEphysDataset, EphysDatasetLabeled, BalancedBatchSampler, none_safe_collate

def get_embeddings_multimodal(loader, model):
    """Extract embeddings from a multimodal model."""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_data = []
    
    with torch.no_grad():
        for sample in loader:
            embedding = model(sample)[0].detach().cpu().numpy()
            # Normalize embeddings
            embedding = (embedding - np.mean(embedding, axis=1, keepdims=True)) / np.std(embedding, axis=1, keepdims=True)
            all_embeddings.extend(embedding)
            label = sample[1]
            if label.ndim == 2:
                cls_label, source_label = label.unbind(1)
            else:
                cls_label = label
            all_labels.extend(cls_label.detach().cpu().numpy())
    
    return np.array(all_embeddings), np.array(all_labels)