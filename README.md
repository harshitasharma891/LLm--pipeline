# RAG Pipeline

## Workflows
1. Update config files
2. Run run.py file



# How to run?
### STEPS:

Clone the repository

```bash
git@gitlab.com:tai_ai_projects/llm_pipeline.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n envname python=3.10.12 -y
```

```bash
conda activate envname
```


### STEP 02- install the requirements
```bash
pip install -r requirement.txt
```


```bash
# To run the pipeline
python run.py
```

```bash
# To run Fast-API 
uvicorn app:app --reload
```
