from pathlib import Path
import pandas as pd

# ---- CONFIG ----

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RUNWAY_IMG_DIRS = [
    DATA_DIR / "runway" / "fashion_runway_images_refined",
    # add more runway datasets here later if needed
]


STREET_IMG_DIRS = [
    DATA_DIR / "street" / "modanet" ,
    # add more street datasets here later if needed
]

OUTPUT_CSV = DATA_DIR / "master_metadata_initial.csv"


def collect_images_from_dirs(dirs, source, original_dataset_name):
    """Scan directories and return a list of dicts with basic metadata."""
    rows = []
    for d in dirs:
        if not d.exists():
            print(f"[WARN] Directory does not exist, skipping: {d}")
            continue

        for img_path in d.rglob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Create a stable image_id
            image_id = f"{source}_{original_dataset_name}_{img_path.stem}"

            rows.append(
                {
                    "image_id": image_id,
                    "source": source,
                    "original_dataset": original_dataset_name,
                    "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def main():
    all_rows = []

    # Runway images
    runway_rows = collect_images_from_dirs(
        RUNWAY_IMG_DIRS,
        source="runway",
        original_dataset_name="fashion_runway_images_refined",
    )
    all_rows.extend(runway_rows)

    # Street images
    street_rows = collect_images_from_dirs(
        STREET_IMG_DIRS,
        source="street",
        original_dataset_name="modanet",
    )
    all_rows.extend(street_rows)

    # Convert to DataFrame
    if not all_rows:
        print("[ERROR] No images found. Check your data directories.")
        return

    df = pd.DataFrame(all_rows)

    # Save CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Saved metadata for {len(df)} images to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
