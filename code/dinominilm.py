

import torch.nn as nn
import torch
from datetime import datetime
import pandas as pd
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import numpy as np

from sentence_transformers import SentenceTransformer


#### MODEL SETUP

class DinoMiniLMDualEncoder(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()

        # Text encoder (MiniLM)
        self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2")

        # Image encoder (DINO)
        self.image_encoder = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")

        # Projection layers
        self.image_projection = nn.Linear(in_features=384, out_features=768)
        self.text_projection = nn.Linear(in_features=384, out_features=768)

        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, path):
        try:
            # Try to get device from existing parameters, fallback to CPU
            device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
            
            state_dict = torch.load(path, map_location=device)
            self.load_state_dict(state_dict)
            print(f"✅ Loaded weights from: {path}")
        except Exception as e:
            print(f"❌ Failed to load weights: {e}")

    def forward(self, image, text):
        device = next(self.parameters()).device  # model's device (cpu/cuda)

        # --- Text encoding ---
        text_inputs = self.text_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        text_outputs = self.text_encoder(**text_inputs)
        text_embeddings = text_outputs.last_hidden_state.mean(dim=1)
        text_embeddings = self.text_projection(text_embeddings)

        # --- Image encoding ---
        image_processed = self.image_processor(images=image, return_tensors="pt").to(device)
        image_outputs = self.image_encoder(**image_processed)
        image_embeddings = self.image_projection(image_outputs.pooler_output)

        return text_embeddings, image_embeddings