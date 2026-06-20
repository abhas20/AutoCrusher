import os
import pickle
from pathlib import Path

MODEL_DIR = Path("./bias_pipeline_output")
DEFAULT_MODEL_PATH = MODEL_DIR / "fair_adult_model.pkl"

# In-memory cache: { model_filename: {"bundle": bundle, "last_modified": timestamp} }
model_cache = {}

def load_model(model_filename: str = "fair_adult_model.pkl"):
    """
    Checks if the model is in the cache and if it is up-to-date on disk.
    If the file on disk is newer, or not yet in cache, it loads it.
    """
    global model_cache
    model_path = MODEL_DIR / model_filename

    if not model_path.exists():
        print(f" Model path {model_path} does not exist on disk.")
        return None

    try:
        # Get the last modified timestamp of the file
        disk_mtime = os.path.getmtime(model_path)
        cached_item = model_cache.get(model_filename)

        # If not cached, or if the file on disk has been updated
        if cached_item is None or disk_mtime > cached_item["last_modified"]:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)

            model_cache[model_filename] = {
                "bundle": bundle,
                "last_modified": disk_mtime
            }
            print(f"Loaded and cached model: {model_filename} (timestamp: {disk_mtime})")
            return bundle

        # Serve from cache
        return cached_item["bundle"]

    except Exception as e:
        print(f"⚠️ Error loading model bundle {model_filename}: {e}")
        return None
