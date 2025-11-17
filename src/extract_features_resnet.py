from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms


# ---------------- CONFIG ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

METADATA_CSV = DATA_DIR / "master_metadata_initial.csv"

# Salidas
FEATURES_NPY = DATA_DIR / "image_features_resnet50.npy"
INDEX_CSV = DATA_DIR / "image_features_index.csv"

# Para pruebas, puedes limitar el número de imágenes:
# Pon un número (ej. 300) para probar más rápido, o None para usar todas
MAX_IMAGES = 300


# ---------------- MODEL & TRANSFORMS ----------------

def get_model_and_transform(device):
    """
    Carga un ResNet50 preentrenado en ImageNet y devuelve el modelo
    sin la capa final (fc) + las transformaciones de entrada.
    Compatible con versiones recientes de torchvision.
    """
    # Pesos por defecto de ResNet50 (ImageNet)
    weights = models.ResNet50_Weights.DEFAULT

    # Cargar el modelo con esos pesos
    base_model = models.resnet50(weights=weights)

    # Quitar la capa fully-connected final para quedarnos con un vector de 2048 dims
    model = nn.Sequential(*list(base_model.children())[:-1])  # todo menos la última capa
    model.eval()
    model.to(device)

    # Transforms recomendados por los pesos (incluye resize, crop, normalización, etc.)
    preprocess = weights.transforms()

    return model, preprocess


# ---------------- FEATURE EXTRACTION ----------------

def extract_features():
    # Cargar metadata
    print(f"[INFO] Loading metadata from {METADATA_CSV}")
    df = pd.read_csv(METADATA_CSV)

    if MAX_IMAGES is not None:
        df = df.sample(min(MAX_IMAGES, len(df)), random_state=42).reset_index(drop=True)
        print(f"[INFO] Using a subset of {len(df)} images for this run.")
    else:
        print(f"[INFO] Using all {len(df)} images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model, preprocess = get_model_and_transform(device)

    all_features = []
    image_ids = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        img_path = PROJECT_ROOT / row["image_path"]
        image_id = row["image_id"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open image {img_path}: {e}")
            continue

        # Preprocesar y pasar por el modelo
        input_tensor = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]

        with torch.no_grad():
            feat = model(input_tensor)  # [1, 2048, 1, 1]
            feat = feat.squeeze().cpu().numpy()  # [2048]

        all_features.append(feat)
        image_ids.append(image_id)

    if not all_features:
        print("[ERROR] No features extracted. Check your image paths.")
        return

    features_array = np.stack(all_features, axis=0)
    print(f"[INFO] Feature array shape: {features_array.shape}")

    # Guardar .npy
    np.save(FEATURES_NPY, features_array)
    print(f"[OK] Saved features to {FEATURES_NPY}")

    # Guardar índice
    index_df = pd.DataFrame({
        "image_id": image_ids,
        "feature_row": list(range(len(image_ids))),
    })
    index_df.to_csv(INDEX_CSV, index=False)
    print(f"[OK] Saved feature index to {INDEX_CSV}")


if __name__ == "__main__":
    extract_features()
