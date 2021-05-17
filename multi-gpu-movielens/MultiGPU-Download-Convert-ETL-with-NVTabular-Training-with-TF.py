# External dependencies
import os
import pathlib

import cudf  # cuDF is an implementation of Pandas-like Dataframe on GPU

from nvtabular.utils import download_file
from sklearn.model_selection import train_test_split

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


# Standard Libraries
import shutil

# External Dependencies
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import rmm

# NVTabular
import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.io import Shuffle
from nvtabular.utils import device_mem_size


# define some information about where to get our data
input_path = pathlib.Path(BASE_DIR, "converted", "movielens")
dask_workdir = pathlib.Path(BASE_DIR, "test_dask", "workdir")
output_path = pathlib.Path(BASE_DIR, "test_dask", "output")
stats_path = pathlib.Path(BASE_DIR, "test_dask", "stats")

# Make sure we have a clean worker space for Dask
if pathlib.Path.is_dir(dask_workdir):
    shutil.rmtree(dask_workdir)
dask_workdir.mkdir(parents=True)

# Make sure we have a clean stats space for Dask
if pathlib.Path.is_dir(stats_path):
    shutil.rmtree(stats_path)
stats_path.mkdir(parents=True)

# Make sure we have a clean output path
if pathlib.Path.is_dir(output_path):
    shutil.rmtree(output_path)
output_path.mkdir(parents=True)

# Get device memory capacity
capacity = device_mem_size(kind="total")



# Deploy a Single-Machine Multi-GPU Cluster
protocol = "tcp"  # "tcp" or "ucx"
visible_devices = "0,1"  # Delect devices to place workers
device_spill_frac = 0.5  # Spill GPU-Worker memory to host at this limit.
# Reduce if spilling fails to prevent
# device memory errors.
cluster = None  # (Optional) Specify existing scheduler port
if cluster is None:
    cluster = LocalCUDACluster(
        protocol=protocol,
        CUDA_VISIBLE_DEVICES=visible_devices,
        local_directory=dask_workdir,
        device_memory_limit=capacity * device_spill_frac,
    )

# Create the distributed client
client = Client(cluster)


# Initialize RMM pool on ALL workers
def _rmm_pool():
    rmm.reinitialize(
        pool_allocator=True,
        initial_pool_size=None,  # Use default size
    )


client.run(_rmm_pool)


movies = cudf.read_parquet(pathlib.Path(BASE_DIR, "ml-25m", "movies_converted.parquet"))
joined = ["userId", "movieId"] >> nvt.ops.JoinExternal(movies, on=["movieId"])
cat_features = joined >> nvt.ops.Categorify()
ratings = nvt.ColumnGroup(["rating"]) >> (lambda col: (col > 3).astype("int8"))
output = cat_features + ratings
# USE client in NVTabular workfow
workflow = nvt.Workflow(output, client=client)
# !rm -rf $BASE_DIR/train
# !rm -rf $BASE_DIR/valid
train_iter = nvt.Dataset([str(pathlib.Path(BASE_DIR, "train.parquet"))], part_size="100MB")
valid_iter = nvt.Dataset([str(pathlib.Path(BASE_DIR, "valid.parquet"))], part_size="100MB")
workflow.fit(train_iter)
workflow.save(pathlib.Path(BASE_DIR, "workflow"))
shuffle = Shuffle.PER_WORKER  # Shuffle algorithm
out_files_per_proc = 4  # Number of output files per worker
workflow.transform(train_iter).to_parquet(
    output_path=pathlib.Path(BASE_DIR, "train"),
    shuffle=shuffle,
    out_files_per_proc=out_files_per_proc,
)
workflow.transform(valid_iter).to_parquet(
    output_path=pathlib.Path(BASE_DIR, "valid"),
    shuffle=shuffle,
    out_files_per_proc=out_files_per_proc,
)

client.shutdown()
cluster.close()


# External dependencies
import argparse
import glob
import os

import cupy

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.3"  # fraction of free memory

import nvtabular as nvt  # noqa: E402 isort:skip
from nvtabular.framework_utils.tensorflow import layers  # noqa: E402 isort:skip
from nvtabular.loader.tensorflow import KerasSequenceLoader  # noqa: E402 isort:skip

import tensorflow as tf  # noqa: E402 isort:skip
import horovod.tensorflow as hvd  # noqa: E402 isort:skip

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--dir_in", default=None, help="Input directory")
parser.add_argument("--batch_size", default=None, help="batch size")
parser.add_argument("--cats", default=None, help="categorical columns")
parser.add_argument("--cats_mh", default=None, help="categorical multihot columns")
parser.add_argument("--conts", default=None, help="continuous columns")
parser.add_argument("--labels", default=None, help="continuous columns")
args = parser.parse_args()


BASE_DIR = args.dir_in or "./data/"
BATCH_SIZE = int(args.batch_size or 16384)  # Batch Size
CATEGORICAL_COLUMNS = args.cats or ["movieId", "userId"]  # Single-hot
CATEGORICAL_MH_COLUMNS = args.cats_mh or ["genres"]  # Multi-hot
NUMERIC_COLUMNS = args.conts or []
TRAIN_PATHS = sorted(
    glob.glob(os.path.join(BASE_DIR, "train/*.parquet"))
)  # Output from ETL-with-NVTabular
hvd.init()

# Seed with system randomness (or a static seed)
cupy.random.seed(None)


def seed_fn():
    """
    Generate consistent dataloader shuffle seeds across workers

    Reseeds each worker's dataloader each epoch to get fresh a shuffle
    that's consistent across workers.
    """
    min_int, max_int = tf.int32.limits
    max_rand = max_int // hvd.size()

    # Generate a seed fragment on each worker
    seed_fragment = cupy.random.randint(0, max_rand).get()

    # Aggregate seed fragments from all Horovod workers
    seed_tensor = tf.constant(seed_fragment)
    reduced_seed = hvd.allreduce(seed_tensor, name="shuffle_seed", op=hvd.mpi_ops.Sum)

    return reduced_seed % max_rand


proc = nvt.Workflow.load(os.path.join(BASE_DIR, "workflow/"))
EMBEDDING_TABLE_SHAPES = nvt.ops.get_embedding_sizes(proc)

train_dataset_tf = KerasSequenceLoader(
    TRAIN_PATHS,  # you could also use a glob pattern
    batch_size=BATCH_SIZE,
    label_names=["rating"],
    cat_names=CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS,
    cont_names=NUMERIC_COLUMNS,
    engine="parquet",
    shuffle=True,
    buffer_size=0.06,  # how many batches to load at once
    parts_per_chunk=1,
    global_size=hvd.size(),
    global_rank=hvd.rank(),
    seed_fn=seed_fn,
)
inputs = {}  # tf.keras.Input placeholders for each feature to be used
emb_layers = []  # output of all embedding layers, which will be concatenated
for col in CATEGORICAL_COLUMNS:
    inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))
# Note that we need two input tensors for multi-hot categorical features
for col in CATEGORICAL_MH_COLUMNS:
    inputs[col + "__values"] = tf.keras.Input(name=f"{col}__values", dtype=tf.int64, shape=(1,))
    inputs[col + "__nnzs"] = tf.keras.Input(name=f"{col}__nnzs", dtype=tf.int64, shape=(1,))
for col in CATEGORICAL_COLUMNS + CATEGORICAL_MH_COLUMNS:
    emb_layers.append(
        tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(
                col, EMBEDDING_TABLE_SHAPES[col][0]
            ),  # Input dimension (vocab size)
            EMBEDDING_TABLE_SHAPES[col][1],  # Embedding output dimension
        )
    )
emb_layer = layers.DenseFeatures(emb_layers)
x_emb_output = emb_layer(inputs)
x = tf.keras.layers.Dense(128, activation="relu")(x_emb_output)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=x)
loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.SGD(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
checkpoint_dir = "./checkpoints"
checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)


@tf.function(experimental_relax_shapes=True)
def training_step(examples, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = model(examples, training=True)
        loss_value = loss(labels, probs)
    # Horovod: add Horovod Distributed GradientTape.
    tape = hvd.DistributedGradientTape(tape, sparse_as_dense=True)
    grads = tape.gradient(loss_value, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if first_batch:
        hvd.broadcast_variables(model.variables, root_rank=0)
        hvd.broadcast_variables(opt.variables(), root_rank=0)
    return loss_value


# Horovod: adjust number of steps based on number of GPUs.
for batch, (examples, labels) in enumerate(train_dataset_tf):
    loss_value = training_step(examples, labels, batch == 0)
    if batch % 100 == 0 and hvd.local_rank() == 0:
        print("Step #%d\tLoss: %.6f" % (batch, loss_value))
hvd.join()

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
if hvd.rank() == 0:
    checkpoint.save(checkpoint_dir)


