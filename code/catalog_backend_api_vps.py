import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
from PIL import Image
import io
import urllib.parse

from dinominilm import DinoMiniLMDualEncoder

# ==== MODEL AND DATA LOADING ====

# Load model on CPU
weights_file = "/root/tinyvlm/models/dino_minilm_stage2v2.pth"
dino_minilm = DinoMiniLMDualEncoder(weights_path=weights_file).to("cpu")
dino_minilm.eval()

# Load embeddings
embeddings_list = torch.load(
    "/root/tinyvlm/data/list_embeddings_stage2v2.pt", map_location="cpu"
)

# Base image directory
BASE_IMAGE_DIR = Path(
    "/root/tinyvlm/data/image-description-marketplace-data/flip_data_vlm/flip_data_vlm"
)

# ==== CLASSES ====

class QueryEncoder:
    def __init__(self, model):
        self.model = model

    def encode_query(self, query, is_image=False):
        with torch.no_grad():
            if not is_image:
                query_embedding = self.model.text_encoder(
                    **self.model.text_tokenizer(query, return_tensors="pt").to("cpu")
                ).last_hidden_state.mean(dim=1)
                query_embedding = self.model.text_projection(query_embedding)
            else:
                query_embedding = self.model.image_encoder(
                    **self.model.image_processor(images=query, return_tensors="pt").to("cpu")
                ).pooler_output
                query_embedding = self.model.image_projection(query_embedding)
        return query_embedding


class TopkFinder:
    def __init__(self, embeddings_list, base_image_dir=None):
        self.device = torch.device("cpu")
        self.embeddings_list = embeddings_list
        self.base_image_dir = base_image_dir

    def find_most_similar(self, query_embedding, top_k=5, image=False):
        query_embedding = F.normalize(query_embedding.detach().to(self.device), dim=1)
        all_embeddings = torch.stack([item["embedding"].to(self.device) for item in self.embeddings_list])
        all_embeddings = F.normalize(all_embeddings, dim=1)
        similarities = (all_embeddings @ query_embedding.T).squeeze(1)

        topk_indices = similarities.topk(top_k).indices.tolist()
        results = []

        for i in topk_indices:
            item = self.embeddings_list[i]
            score = similarities[i].item()
            if image and self.base_image_dir:
                parts = Path(item["image_path"]).parts
                if "flip_data_vlm" in parts:
                    idx = len(parts) - 1 - parts[::-1].index("flip_data_vlm")
                    relative_path = Path(*parts[idx + 1:])
                    encoded = urllib.parse.quote(str(relative_path))
                    url = f"http://159.89.15.19:8000/images/{encoded}"
                    results.append({"url": url, "similarity": score})
                else:
                    url = f"http://159.89.15.19:8000/images/{urllib.parse.quote(Path(item['image_path']).name)}"
                    results.append({"url": url, "similarity": score})
            elif not image:
                results.append({"text": item["text"], "similarity": score})
        return results


# ==== INITIALIZE ====

query_encoder = QueryEncoder(dino_minilm)
topk_finder = TopkFinder(embeddings_list, base_image_dir=BASE_IMAGE_DIR)

# ==== FASTAPI APP ====

app = FastAPI()

# Serve image files
app.mount("/images", StaticFiles(directory=str(BASE_IMAGE_DIR)), name="images")


@app.get("/")
async def health_check():
    return {"status": "API is running", "model_loaded": True}


@app.get("/list_images")
def list_images():
    """Return all image URLs available on the server."""
    image_paths = []
    for pattern in ["category_*/images/*.jpg", "flo_images/*.jpg"]:
        for p in BASE_IMAGE_DIR.glob(pattern):
            encoded_path = urllib.parse.quote(str(p.relative_to(BASE_IMAGE_DIR)))
            full_url = f"http://159.89.15.19:8000/images/{encoded_path}"
            image_paths.append(full_url)
    return JSONResponse(content={"images": image_paths})


@app.post("/encode_image_query")
async def encode_image_query(image_query: UploadFile = File(...)):
    try:
        content = await image_query.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_encoded = query_encoder.encode_query(query=image, is_image=True)
        topk_similar_items = topk_finder.find_most_similar(
            query_embedding=image_encoded, top_k=5, image=True
        )
        return {"top_k_similar_items": topk_similar_items}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/encode_text_query")
async def encode_text_query(text_query: str):
    try:
        text_encoded = query_encoder.encode_query(query=text_query, is_image=False)
        topk_similar_items = topk_finder.find_most_similar(
            query_embedding=text_encoded, top_k=5, image=True
        )
        return {"top_k_similar_items": topk_similar_items}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ==== MAIN ENTRYPOINT ====

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
