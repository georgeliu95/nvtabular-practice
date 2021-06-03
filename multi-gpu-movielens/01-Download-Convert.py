# External dependencies
import os
import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU

from sklearn.model_selection import train_test_split

from nvtabular.utils import download_file


INPUT_DATA_DIR = os.environ.get(
    "INPUT_DATA_DIR", os.path.expanduser("~/nvt-examples/multigpu-movielens/data/")
)


download_file(
    "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
    os.path.join(INPUT_DATA_DIR, "ml-25m.zip"),
)


movies = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "ml-25m/movies.csv"))
movies.head()


movies["genres"] = movies["genres"].str.split("|")
movies = movies.drop("title", axis=1)
movies.head()


movies.to_parquet(os.path.join(INPUT_DATA_DIR, "movies_converted.parquet"))


ratings = cudf.read_csv(os.path.join(INPUT_DATA_DIR, "ml-25m", "ratings.csv"))
ratings.head()


ratings = ratings.drop("timestamp", axis=1)
train, valid = train_test_split(ratings, test_size=0.2, random_state=42)


train.to_parquet(os.path.join(INPUT_DATA_DIR, "train.parquet"))
valid.to_parquet(os.path.join(INPUT_DATA_DIR, "valid.parquet"))
