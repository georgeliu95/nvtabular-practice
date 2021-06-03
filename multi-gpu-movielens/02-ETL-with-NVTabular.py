# External dependencies
import os
import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU
import shutil
import numpy as np

import nvtabular as nvt

from os import path


INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/movielens/data/")
)


movies = cudf.read_parquet(os.path.join(INPUT_DATA_DIR, "movies_converted.parquet"))
movies.head()


CATEGORICAL_COLUMNS = ["userId", "movieId"]
LABEL_COLUMNS = ["rating"]


joined = ["userId", "movieId"] >> nvt.ops.JoinExternal(movies, on=["movieId"])


cat_features = joined >> nvt.ops.Categorify()


ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))


output = cat_features + ratings


workflow = nvt.Workflow(output)


dict_dtypes = {}

for col in CATEGORICAL_COLUMNS:
    dict_dtypes[col] = np.int64

for col in LABEL_COLUMNS:
    dict_dtypes[col] = np.float32


train_dataset = nvt.Dataset([os.path.join(INPUT_DATA_DIR, "train.parquet")], part_size="100MB")
valid_dataset = nvt.Dataset([os.path.join(INPUT_DATA_DIR, "valid.parquet")], part_size="100MB")


workflow.fit(train_dataset)


# Make sure we have a clean output path
if path.exists(os.path.join(INPUT_DATA_DIR, "train")):
    shutil.rmtree(os.path.join(INPUT_DATA_DIR, "train"))
if path.exists(os.path.join(INPUT_DATA_DIR, "valid")):
    shutil.rmtree(os.path.join(INPUT_DATA_DIR, "valid"))

out_files_per_proc = 4  # Number of output files per worker
workflow.transform(train_dataset).to_parquet(
    output_path=os.path.join(INPUT_DATA_DIR, "train"),
    shuffle=nvt.io.Shuffle.PER_PARTITION,
    cats=["userId", "movieId"],
    labels=["rating"],
    dtypes=dict_dtypes,
    out_files_per_proc=out_files_per_proc,
)


workflow.transform(valid_dataset).to_parquet(
    output_path=os.path.join(INPUT_DATA_DIR, "valid"),
    shuffle=False,
    cats=["userId", "movieId"],
    labels=["rating"],
    dtypes=dict_dtypes,
    out_files_per_proc=out_files_per_proc,
)


workflow.save(os.path.join(INPUT_DATA_DIR, "workflow"))


import glob

TRAIN_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "train", "*.parquet")))
VALID_PATHS = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "valid", "*.parquet")))
TRAIN_PATHS, VALID_PATHS


df = cudf.read_parquet(TRAIN_PATHS[0])
df.head()

