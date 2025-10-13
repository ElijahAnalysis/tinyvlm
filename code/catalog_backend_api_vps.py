import torch
import torch.nn.functional as F

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from dinominilm import DinoMiniLMDualEncoder

from pathlib import Path
from PIL import Image

import io


# Load model
weights_file = "/root/tinyvlm/models/dino_minilm_stage2v2.pth"
dino_minilm = DinoMiniLMDualEncoder(weights_path=weights_file).to('cpu')
dino_minilm.eval()


# Load embeddings
embeddings_list = torch.load("/root/tinyvlm/data/list_embeddings_stage2v2.pt")


#### CLASSES

class QueryEncoder:
    def __init__(self, model):
        self.model = model

    def encode_query(self, query, is_image=False):
        with torch.no_grad():  # Add no_grad for inference
            if is_image == False:
                query_embedding = self.model.text_encoder(
                    **self.model.text_tokenizer(query, return_tensors="pt").to('cuda')
                ).last_hidden_state.mean(dim=1)
                query_embedding = self.model.text_projection(query_embedding)
            else:
                query_embedding = self.model.image_encoder(
                    **self.model.image_processor(images=query, return_tensors="pt").to('cuda')
                ).pooler_output
                query_embedding = self.model.image_projection(query_embedding)

        return query_embedding


class TopkFinder:
    def __init__(self, embeddings_list, base_image_dir=None):
        self.device = torch.device("cuda")
        self.embeddings_list = embeddings_list
        self.base_image_dir = base_image_dir

    def find_most_similar(self, query_embedding, top_k=5, image=False):
        """
        Find most similar items (images or texts) given a query embedding.

        Args:
            query_embedding: Tensor of shape [1, dim]
            embeddings_list: list of dicts with keys:
                {"image_path", "embedding"} if image=True
                {"text", "embedding"} if image=False
            top_k: number of top matches to return
            image: whether we're searching among image embeddings (bool)
        """
        # Move query to device
        query_embedding = F.normalize(query_embedding.detach().to(self.device), dim=1)

        # Stack and normalize embeddings
        all_embeddings = torch.stack([item["embedding"].to(self.device) for item in self.embeddings_list])
        all_embeddings = F.normalize(all_embeddings, dim=1)

        # Compute cosine similarities
        similarities = (all_embeddings @ query_embedding.T).squeeze(1)

        # Top-k indices
        topk_indices = similarities.topk(top_k).indices.tolist()

        # Retrieve results
        if image:
            raw_paths = [self.embeddings_list[i]["image_path"] for i in topk_indices]
            # Fix paths: extract filename and combine with base_image_dir
            if self.base_image_dir:
                top_items = []
                for path in raw_paths:
                    # Extract the relative path after flip_data_vlm/flip_data_vlm/
                    parts = Path(path).parts
                    if 'flip_data_vlm' in parts:
                        # Find the last occurrence of flip_data_vlm and take everything after it
                        idx = len(parts) - 1 - parts[::-1].index('flip_data_vlm')
                        relative_path = Path(*parts[idx+1:])
                        fixed_path = str(Path(self.base_image_dir) / relative_path)
                    else:
                        # Fallback: just use the filename
                        fixed_path = str(Path(self.base_image_dir) / Path(path).name)
                    top_items.append(fixed_path)
            else:
                top_items = raw_paths
        else:
            top_items = [self.embeddings_list[i]["text"] for i in topk_indices]

        top_scores = [similarities[i].item() for i in topk_indices]

        return list(zip(top_items, top_scores))


# Initialize encoder and finder
base_image_directory = "/root/tinyvlm/data/image-description-marketplace-data/flip_data_vlm/flip_data_vlm"
query_encoder = QueryEncoder(dino_minilm)
topk_finder = TopkFinder(embeddings_list, base_image_dir=base_image_directory)


### API

app = FastAPI()

# Mount static files to serve images
app.mount("/images", StaticFiles(directory="/root/tinyvlm/data/image-description-marketplace-data/flip_data_vlm/flip_data_vlm"), name="images")


@app.post('/encode_image_query')
async def encode_image_query(image_query: UploadFile = File(...)):
    try:
        # Read image content
        content = await image_query.read()  # Fixed: was 'image' should be 'image_query'
        image = Image.open(io.BytesIO(content)).convert("RGB")

        # Encode image
        image_encoded = query_encoder.encode_query(query=image, is_image=True)

        # Find similar items
        topk_similar_items = topk_finder.find_most_similar(
            query_embedding=image_encoded, top_k=5, image=True
        )
        
        return {'top_k_similar_items': topk_similar_items}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post('/encode_text_query')
async def encode_text_query(text_query: str):
    try:
        # Encode text
        text_encoded = query_encoder.encode_query(query=text_query, is_image=False)

        # Find similar items
        topk_similar_items = topk_finder.find_most_similar(
            query_embedding=text_encoded, top_k=5, image=True
        )
        
        return {'top_k_similar_items': topk_similar_items}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})


# Optional: Add a health check endpoint
@app.get('/')
async def health_check():
    return {'status': 'API is running', 'model_loaded': True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)