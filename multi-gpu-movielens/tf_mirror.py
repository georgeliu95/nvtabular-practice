import tensorflow as tf
import pandas as pd

# External dependencies
import argparse
import glob
import os

from nvtabular.framework_utils.tensorflow import layers  # noqa: E402 isort:skip
import numpy as np

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.3"  # fraction of free memory


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
EMBEDDING_TABLE_SHAPES = {'movieId': (56586, 512), 'userId': (162542, 512), 'genres': (21, 16)}
TRAIN_PATHS = sorted(
    glob.glob(os.path.join(BASE_DIR, "train/*.parquet"))
)  # Output from ETL-with-NVTabular


# dataframe = pd.read_parquet(TRAIN_PATHS[0])
# dataframe = pd.concat([pd.read_parquet(it) for it in TRAIN_PATHS])
df_list = []
for it in TRAIN_PATHS:
    tmp = pd.read_parquet(it)
    tmp.pop("genres")
    df_list.append(tmp)
dataframe = pd.concat(df_list)


mirrored_strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))


inputs = {}  # tf.keras.Input placeholders for each feature to be used
emb_layers = []  # output of all embedding layers, which will be concatenated
with mirrored_strategy.scope():
    for col in CATEGORICAL_COLUMNS:
        inputs[col] = tf.keras.Input(name=col, dtype=tf.int32, shape=(1,))
    for col in CATEGORICAL_COLUMNS:
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
    x = tf.keras.layers.BatchNormalization()(x_emb_output)
    x = tf.keras.layers.Dense(4096, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(1024, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(64, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    opt = tf.keras.optimizers.Adam(0.001)
    # loss = tf.losses.BinaryCrossentropy()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,  reduction=tf.keras.losses.Reduction.NONE)

def create_dataframe_dict(dataframe):
    df = dataframe.copy()
    labels = df.pop("rating")
    # genres = df.pop("genres")
    dataframe_dict = dict(df)
    for it in dataframe_dict:
        dataframe_dict[it] = dataframe_dict[it].ravel().reshape(-1,1)
    labels = labels.ravel().reshape(-1,1)
    # dataframe_dict["genres"] = tf.ragged.constant(genres.ravel())
    return dataframe_dict, labels

ds = tf.data.Dataset.from_tensor_slices(create_dataframe_dict(dataframe))

ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(ds)


with mirrored_strategy.scope():
    @tf.function(experimental_relax_shapes=True)
    def training_step(inputs):
        examples, labels = inputs
        with tf.GradientTape() as tape:
            probs = model(examples, training=True)
            print(type(loss))
            loss_value = loss(labels, probs)
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value


    @tf.function
    def distributed_train_step(inputs):
        per_replica_losses = mirrored_strategy.run(training_step, args=(inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    for batch, (example, label) in enumerate(dist_dataset):
        # if batch ==0:
        #     print(example)
        loss_val = distributed_train_step((example, label))
        if batch % 100 ==0:
            print("Step #%d\tLoss: " % (batch), loss_val)

