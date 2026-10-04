# Small Defect Dataset Explorer

This folder contains a small CLI and Streamlit dashboard for summarizing the processed dataset.

## Export CSV/JSON summaries

```bash
python dataset_explorer/dataset_query_tool.py export
```

Outputs:

- `dataset_explorer/outputs/dataset_summary.csv`
- `dataset_explorer/outputs/dataset_class_size_summary.csv`
- `dataset_explorer/outputs/dataset_summary.json`

## Ask quick dataset questions

```bash
python dataset_explorer/dataset_query_tool.py datasets
python dataset_explorer/dataset_query_tool.py surfaces
python dataset_explorer/dataset_query_tool.py categories
python dataset_explorer/dataset_query_tool.py counts
python dataset_explorer/dataset_query_tool.py aggregate
python dataset_explorer/dataset_query_tool.py counts --dataset GC10-DET
python dataset_explorer/dataset_query_tool.py counts --dataset GC10-DET --class-name 1_chongkong
```

## Run the dashboard

```bash
streamlit run dataset_explorer/app.py
```

The dashboard shows:

- overall small/medium/large counts
- dataset-wise counts
- surface types
- listed class names and translated meanings
- class-wise small/medium/large counts
- filters by dataset, label type, size, and text search
