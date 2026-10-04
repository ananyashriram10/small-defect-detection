from pathlib import Path

import pandas as pd
import streamlit as st

from dataset_query_tool import build_summary, export_outputs


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_path(processed_root, dataset, size_folder, image_file):
    image_path = processed_root / dataset / size_folder / "images" / image_file
    if image_path.exists():
        return image_path
    fallback = processed_root / dataset / size_folder / image_file
    return fallback if fallback.exists() else None


def find_mask_path(processed_root, dataset, size_folder, image_file):
    stem = Path(image_file).stem
    candidates = [
        processed_root / dataset / size_folder / "masks" / f"{stem}_mask.png",
        processed_root / dataset / size_folder / "masks" / f"{stem.replace('_defect', '')}_mask.png",
        processed_root / dataset / size_folder / "masks" / f"{stem.replace('_defect', '_mask')}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    masks_dir = processed_root / dataset / size_folder / "masks"
    if masks_dir.exists():
        prefix = stem.replace("_defect", "")
        matches = sorted(masks_dir.glob(f"{prefix}*"))
        if matches:
            return matches[0]
    return None


def load_manifest(processed_root, dataset):
    manifest_path = processed_root / dataset / "size_manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame()
    return pd.read_csv(manifest_path)


st.set_page_config(page_title="Small Defect Dataset Explorer", layout="wide")

st.title("Small Defect Dataset Explorer")

processed_root = Path(st.sidebar.text_input("Processed dataset root", "processed_output"))
output_dir = Path(st.sidebar.text_input("Export output folder", "dataset_explorer/outputs"))

dataset_summary, class_summary, aggregate = build_summary(processed_root)

if st.sidebar.button("Export CSV/JSON summaries"):
    export_outputs(dataset_summary, class_summary, aggregate, output_dir)
    st.sidebar.success(f"Saved summaries to {output_dir}")

st.subheader("Overall Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total images", f"{aggregate['total']:,}")
c2.metric("Small", f"{aggregate['small']:,}")
c3.metric("Medium", f"{aggregate['medium']:,}")
c4.metric("Large", f"{aggregate['large']:,}")

size_df = pd.DataFrame(
    [
        {"size": "small", "images": aggregate["small"]},
        {"size": "medium", "images": aggregate["medium"]},
        {"size": "large", "images": aggregate["large"]},
    ]
)

left, right = st.columns([1, 2])
with left:
    st.caption("Small / medium / large split")
    st.bar_chart(size_df.set_index("size"))
with right:
    st.caption("Dataset image counts")
    st.bar_chart(dataset_summary.set_index("dataset")[["small", "medium", "large"]])

st.subheader("Filters")
fc1, fc2, fc3, fc4 = st.columns(4)

dataset_options = ["All"] + dataset_summary["dataset"].tolist()
dataset_choice = fc1.selectbox("Dataset", dataset_options)

label_options = ["All"] + sorted(class_summary["label_type"].dropna().unique().tolist())
label_choice = fc2.selectbox("Label type", label_options)

size_choice = fc3.selectbox("Size column to inspect", ["all", "small", "medium", "large"])
search = fc4.text_input("Class/meaning contains")

filtered = class_summary.copy()

if dataset_choice != "All":
    filtered = filtered[filtered["dataset"] == dataset_choice]

if label_choice != "All":
    filtered = filtered[filtered["label_type"] == label_choice]

if search:
    mask = (
        filtered["listed_class"].str.contains(search, case=False, na=False)
        | filtered["translated_class"].str.contains(search, case=False, na=False)
        | filtered["surface_type"].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

if size_choice in {"small", "medium", "large"}:
    filtered = filtered[filtered[size_choice] > 0]

st.subheader("Filtered Count Summary")
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Rows", f"{len(filtered):,}")
sc2.metric("Small images", f"{int(filtered['small'].sum()):,}")
sc3.metric("Medium images", f"{int(filtered['medium'].sum()):,}")
sc4.metric("Large images", f"{int(filtered['large'].sum()):,}")

st.subheader("Dataset Summary")
st.dataframe(
    dataset_summary[
        ["dataset", "surface_type", "label_type", "num_classes", "small", "medium", "large", "total"]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Class / Category Summary")
st.dataframe(
    filtered[
        [
            "dataset",
            "surface_type",
            "listed_class",
            "translated_class",
            "label_type",
            "small",
            "medium",
            "large",
            "total",
        ]
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Sample Images And Masks")
if dataset_choice == "All":
    st.info("Select one dataset above to preview image and mask samples.")
else:
    manifest = load_manifest(processed_root, dataset_choice)
    if manifest.empty:
        st.warning("No manifest found for this dataset.")
    else:
        class_options = ["All"] + sorted(manifest["class_name"].dropna().astype(str).unique().tolist())
        pc1, pc2, pc3 = st.columns(3)
        preview_class = pc1.selectbox("Preview class", class_options)
        preview_size = pc2.selectbox("Preview size", ["all", "small", "medium", "large"])
        preview_count = pc3.slider("Samples", min_value=1, max_value=12, value=4)

        preview_df = manifest.copy()
        if preview_class != "All":
            preview_df = preview_df[preview_df["class_name"].astype(str) == preview_class]
        if preview_size != "all":
            preview_df = preview_df[preview_df["size_folder"] == preview_size]

        preview_df = preview_df.drop_duplicates(["image_file", "size_folder"]).head(preview_count)

        if preview_df.empty:
            st.warning("No samples match the selected preview filters.")
        else:
            cols = st.columns(2)
            for idx, row in enumerate(preview_df.to_dict(orient="records")):
                image_path = find_image_path(
                    processed_root,
                    dataset_choice,
                    row["size_folder"],
                    row["image_file"],
                )
                mask_path = find_mask_path(
                    processed_root,
                    dataset_choice,
                    row["size_folder"],
                    row["image_file"],
                )

                with cols[idx % 2]:
                    st.caption(
                        f"{row['image_file']} | class: {row['class_name']} | size: {row['size_folder']}"
                    )
                    if image_path:
                        st.image(str(image_path), caption="image", use_container_width=True)
                    else:
                        st.warning("Image file not found.")
                    if mask_path:
                        st.image(str(mask_path), caption="pixel-level mask", use_container_width=True)
                    else:
                        st.caption("Mask preview not found for this sample.")

st.subheader("Quick Questions")
question = st.selectbox(
    "Choose a question",
    [
        "What datasets do we have?",
        "What surfaces are covered?",
        "What categories/classes are in each dataset?",
        "What is the aggregate small/medium/large split?",
    ],
)

if question == "What datasets do we have?":
    st.write(", ".join(dataset_summary["dataset"].tolist()))
elif question == "What surfaces are covered?":
    st.dataframe(dataset_summary[["dataset", "surface_type"]], hide_index=True, use_container_width=True)
elif question == "What categories/classes are in each dataset?":
    st.dataframe(
        class_summary[["dataset", "listed_class", "translated_class", "label_type"]],
        hide_index=True,
        use_container_width=True,
    )
else:
    st.write(
        {
            "small": aggregate["small"],
            "medium": aggregate["medium"],
            "large": aggregate["large"],
            "total": aggregate["total"],
        }
    )
