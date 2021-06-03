# External dependencies
import os
import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU
import pathlib

from sklearn.model_selection import train_test_split

from nvtabular.utils import download_file


INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", "~/nvt-examples/multigpu-movielens/data/"
)
BASE_DIR = pathlib.Path(INPUT_DATA_DIR).expanduser()
zip_path = pathlib.Path(BASE_DIR, "ml-25m.zip")
download_file(
    "http://files.grouplens.org/datasets/movielens/ml-25m.zip", zip_path, redownload=False
)


movies = cudf.read_csv(pathlib.Path(BASE_DIR, "ml-25m", "movies.csv"))
movies["genres"] = movies["genres"].str.split("|")
movies = movies.drop("title", axis=1)
movies.to_parquet(pathlib.Path(BASE_DIR, "ml-25m", "movies_converted.parquet"))


ratings = cudf.read_csv(pathlib.Path(BASE_DIR, "ml-25m", "ratings.csv"))
ratings = ratings.drop("timestamp", axis=1)
train, valid = train_test_split(ratings, test_size=0.2, random_state=42)
train.to_parquet(pathlib.Path(BASE_DIR, "train.parquet"))
valid.to_parquet(pathlib.Path(BASE_DIR, "valid.parquet"))
