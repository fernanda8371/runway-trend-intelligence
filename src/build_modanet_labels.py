from pathlib import Path
import json
import pandas as pd

# -------- CONFIG --------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Ajusta el split si hace falta ("train", "valid", etc.)
MODANET_TRAIN_DIR = DATA_DIR / "street" / "modanet" / "train"
COCO_ANN_PATH = MODANET_TRAIN_DIR / "_annotations.coco.json"

OUTPUT_CSV = DATA_DIR / "modanet_labels.csv"


def load_coco_annotations(path: Path):
    print(f"[INFO] Loading COCO annotations from {path}")
    with open(path, "r") as f:
        coco = json.load(f)
    return coco


def build_category_maps(coco):
    # category_id -> name
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    return cat_id_to_name


def build_image_maps(coco):
    # image_id -> file_name
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    return img_id_to_file


def assign_coarse_category(categories_for_image):
    """
    categories_for_image: set de nombres de categoría Modanet (bag, belt, dress, pants, etc.)

    Definimos una categoría principal por imagen, basado en prioridad.
    Puedes ajustar esta lógica si quieres.
    """
    cats = set(categories_for_image)

    # Reglas de prioridad (muy simplificadas)
    if "dress" in cats:
        return "dress"
    if "outer" in cats:
        return "outerwear"
    # top + (pants/shorts/skirt) -> outfit completo
    if "top" in cats and any(c in cats for c in ["pants", "shorts", "skirt"]):
        return "top+bottom"
    if any(c in cats for c in ["pants", "shorts", "skirt"]):
        return "bottoms"
    if "top" in cats:
        return "top"
    if any(c in cats for c in ["boots", "footwear"]):
        return "footwear"
    if "bag" in cats:
        return "bag"

    # accesorios / otros
    if any(c in cats for c in ["belt", "sunglasses", "headwear", "scarf & tie"]):
        return "accessory"

    return "other"


def main():
    if not COCO_ANN_PATH.exists():
        print(f"[ERROR] COCO annotations not found at {COCO_ANN_PATH}")
        return

    coco = load_coco_annotations(COCO_ANN_PATH)
    cat_id_to_name = build_category_maps(coco)
    img_id_to_file = build_image_maps(coco)

    # image_file -> set of category names present
    img_file_to_cats = {}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]

        img_file = img_id_to_file.get(img_id)
        cat_name = cat_id_to_name.get(cat_id)

        if img_file is None or cat_name is None:
            continue

        if img_file not in img_file_to_cats:
            img_file_to_cats[img_file] = set()
        img_file_to_cats[img_file].add(cat_name)

    rows = []
    for file_name, cats in img_file_to_cats.items():
        coarse = assign_coarse_category(cats)
        rows.append(
            {
                "image_filename": file_name,
                "modanet_categories": ",".join(sorted(cats)),
                "coarse_category": coarse,
            }
        )

    df = pd.DataFrame(rows)
    print(f"[INFO] Built labels for {len(df)} images.")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved Modanet labels to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
