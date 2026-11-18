# BigData2026 Assignment 2

This project predicts which of six GitHub activities a user will perform during the following week.

## Data Source

The data used in this project is sourced from [GH Archive](https://www.gharchive.org/), a publicly available archive of GitHub events.

## Data Preprocessing

Download the required hourly GH Archive files into a raw data folder.

Run the preprocessing script provided with the assignment:

```bash
python assignment2/scripts/preprocess.py --raw RAW_DATA_FOLDER --out assignment2/data/db
```

The provided preprocessing script generates the eight required tables used by our model.

## Running the Code

After preprocessing the data, run the notebooks in the following order.

### 1. Feature Engineering

Run:

```text
src/feature_engineering.ipynb
```

This notebook constructs the features and labels used for model training. The feature generation functions are implemented in:

```text
src/features.py
```

### 2. Model Training

Run:

```text
src/model_training.ipynb
```

This notebook trains and evaluates the models using the generated features.

### 3. Test Prediction

Run:

```text
src/test.ipynb
```

This notebook generates predictions for the provided test users.

Run all notebook cells in order from top to bottom.

## Output

The final prediction file follows the required format:

```text
UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push
```

Each activity prediction is either `0` or `1`.

## Evaluation

The format of the final submission can be checked using the provided evaluation script:

```bash
python assignment2/scripts/evaluate.py \
  --pred output_{studentID}.csv \
  --input assignment2/submission/test_input.csv
```
