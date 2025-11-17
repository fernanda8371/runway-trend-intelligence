# Runway–to–Street Fashion Intelligence

0. Why this matters

Runway images represent what brands and designers propose.
Street-style (ModaNet) represents what people actually wear.

- By training on street labels and applying predictions to runway and street images, this project builds a small fashion-intelligence layer:

- It turns unstructured images into structured signals ("dress", "outerwear", "top+bottom", etc.).

- It enables basic trend analysis: e.g., comparing silhouette usage between runway and real-life outfits.

This is a first step toward richer tasks like:

- Predicting styling patterns per designer or season.
- Understanding adoption gaps between runway trends and everyday outfits. 
- Building more advanced fashion recommendation or trend forecasting tools. 


This project is a small **fashion–tech pipeline** that turns runway and street-style images into a **structured dataset** for analysis.

# The goal is to build an **end-to-end system** that:

- Ingests runway + street-style images  
- Extracts visual features using a pre-trained CNN (ResNet50)  
- Learns clothing categories from street-style labels (ModaNet)  
- Applies those predictions to all images  
- Outputs a CSV that can be used for **fashion analysis and trend comparison** between runway and real life.


## 1. Datasets

The project uses two main datasets:

### Runway (high fashion)

- Source: **Roboflow** dataset (COCO format), folder:  
  `data/runway/fashion_runway_images_refined/`
- Contains images of **runway looks** (train/valid/test splits).
- Used as the “**creative / editorial**” side of fashion.

### Street-style (real life)

- Source: **ModaNet** (Roboflow COCO export), folder:  
  `data/street/modanet/`
- Street fashion images with polygon/box annotations and **13 clothing categories**  
  (bag, belt, boots, outer, dress, pants, top, shorts, skirt, footwear, etc.).
- Used as a proxy for **real-life outfits** and to learn clothing categories.

> ⚠️ Note: raw images are **not** tracked in git. The `data/` folder is ignored.


## 2. Project Pipeline

End-to-end pipeline:

1. **Build metadata** – index all images in a single CSV  
   Script: `src/build_initial_metadata.py`  
   Output: `data/master_metadata_initial.csv`  
   - Columns: `image_id`, `source` (`runway`/`street`), `original_dataset`, `image_path`.

2. **Extract visual features (embeddings)**  
   Script: `src/extract_features_resnet.py`  
   - Uses a **ResNet50** pre-trained on ImageNet (TorchVision).
   - Removes the final classification layer and uses the 2048-dim penultimate layer as an embedding.
   - Processes images listed in `master_metadata_initial.csv`.

   Outputs:
   - `data/image_features_resnet50.npy` → shape `(N_images, 2048)`  
   - `data/image_features_index.csv` → `image_id` ↔ `feature_row` mapping

3. **Build labels from ModaNet (street)**  
   Script: `src/build_modanet_labels.py`  
   - Reads `data/street/modanet/train/_annotations.coco.json`  
   - Aggregates all categories per image  
   - Maps them to a **coarse category**:

     ```text
     dress                  → "dress"
     outer                  → "outerwear"
     top + bottoms          → "top+bottom"
     pants/shorts/skirt only→ "bottoms"
     boots/footwear         → "footwear"
     bag                    → "bag"
     accessories            → "accessory"
     other                  → "other"
     ```

   Output: `data/modanet_labels.csv`  
   (columns: `image_filename`, `modanet_categories`, `coarse_category`)

4. **Train category classifier (street embeddings)**  
   Notebook: `notebooks/02_features_check_and_merge.ipynb`

   - Join:
     - `master_metadata_initial.csv`
     - `image_features_index.csv`
     - `image_features_resnet50.npy`
     - `modanet_labels.csv` (only for `source == "street"`)
   - Build training data:
     - `X` = ResNet50 features for street images  
     - `y` = `coarse_category` labels from ModaNet  
   - Train a **Logistic Regression** classifier (scikit-learn) on top of the embeddings.  
   - Evaluate with a standard `classification_report`.

5. **Predict clothing categories for all images**  

   Using the trained classifier:

   - Take all images with features (`merged_df`).  
   - Predict `pred_category` for each image.  
   - Save the final enriched dataset:

     ```text
     data/master_metadata_with_category.csv
     ```

     with columns like:

     ```csv
     image_id,source,original_dataset,image_path,feature_row,pred_category
     runway_0001,runway,Fashion Runway Images,data/runway/.../img1.jpg,0,outerwear
     street_0123,street,modanet,data/street/.../000123.jpg,134,top+bottom
     ...
     ```


## 3. Example: image → row in CSV

The full path from image to structured fashion data looks like this:

```python
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# 1. Load metadata + features + index
meta = pd.read_csv(DATA_DIR / "master_metadata_initial.csv")
features = np.load(DATA_DIR / "image_features_resnet50.npy")
index_df = pd.read_csv(DATA_DIR / "image_features_index.csv")

# 2. Join them
merged = meta.merge(index_df, on="image_id", how="inner")

# 3. Example row
row = merged.sample(1).iloc[0]
print("Image path:", row["image_path"])
print("Predicted category:", row.get("pred_category", "<not yet computed>"))
