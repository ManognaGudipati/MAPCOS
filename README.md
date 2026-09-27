# MAPCOS — Multimodal Multi-Agent Diagnosis of PCOS with Differential Exclusion

## Structure

    mapcos/
    ├── config.py               # paths, thresholds, model file names
    ├── main.py                 # entry point — wires agents into the orchestrator
    ├── agents/                 # one file per agent
    ├── orchestrator/           # Master Orchestrator (dispatch + pipeline)
    ├── knowledge_graph/        # Neo4j client for guideline rules
    ├── utils/                  # shared helpers (Grad-CAM, IoU, ...)
    ├── training/               # training scripts, one per trainable agent
    └── reports/                # builds the Final Clinical Report

## Running in a Kaggle Notebook

1. Add this repo's two datasets via "Add Data" in the notebook (search Kaggle
   for each dataset by name — do not try to clone them, only code lives here).
2. Clone this repo:

       !git clone https://github.com/<your-username>/<your-repo>.git
       %cd <your-repo>

3. Install dependencies:

       !pip install -r requirements.txt

4. Check `mapcos/config.py` — update `DATA_DIR_VISION` / `DATA_DIR_TABULAR`
   to match the actual mounted paths under `/kaggle/input/`.
5. Run training / the pipeline as instructed in each step as it's added.

## Status

Skeleton only — agents are added incrementally, one commit at a time.
