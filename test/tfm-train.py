import tensorflow as tf
import pandas as pd
import os
import time

from nvtabular.framework_utils.tensorflow import layers  # noqa: E402 isort:skip
import numpy as np

import nvtx
from create_data import STEPS, GLOBAL_BATCH_SIZE, EMBEDDING_SIZE, FEATURE_COLUMNS, LABEL_COLUMNS

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.6"  # fraction of free memory


mirrored_strategy = tf.distribute.MirroredStrategy()
# mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))


inputs = {}  # tf.keras.Input placeholders for each feature to be used
emb_layers = []  # output of all embedding layers, which will be concatenated
with mirrored_strategy.scope():
    for col in FEATURE_COLUMNS:
        inputs[col] = tf.keras.Input(name=col, dtype=tf.int64, shape=(1,))
    for col in FEATURE_COLUMNS:
        emb_layers.append(
            tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(
                    col, GLOBAL_BATCH_SIZE+1
                ),  # Input dimension (vocab size)
                EMBEDDING_SIZE,  # Embedding output dimension
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
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,  reduction=tf.keras.losses.Reduction.NONE)

def create_dataframe_dict(df):
    labels = df.pop(LABEL_COLUMNS[0])
    dataframe_dict = dict(df)
    for it in dataframe_dict:
        dataframe_dict[it] = dataframe_dict[it].ravel().reshape(-1,1)
    labels = labels.ravel().reshape(-1,1).astype(np.float32)
    return dataframe_dict, labels

ds = tf.data.Dataset.from_tensor_slices(create_dataframe_dict(pd.read_parquet("./data/train.parquet")))





# -------------------------------------------------------------------------------------------------- #

# batch_size = input_context.get_per_replica_batch_size(BATCH_SIZE)
ds = ds.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
dist_dataset = mirrored_strategy.experimental_distribute_dataset(ds)


with mirrored_strategy.scope():
    def compute_loss(labels, predictions):
        per_example_loss = loss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    @tf.function(experimental_relax_shapes=True)
    def training_step(inputs):
        examples, labels = inputs
        with tf.GradientTape() as tape:
            probs = model(examples, training=True)
            # print(type(loss))
            loss_value = compute_loss(labels, probs)
        grads = tape.gradient(loss_value, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss_value


    @tf.function
    def distributed_train_step(inputs):
        per_replica_losses = mirrored_strategy.run(training_step, args=(inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    train_time = 0
    rng = nvtx.start_range(message="Training phase")
    for batch, (example, label) in enumerate(dist_dataset):
        for i in range(mirrored_strategy.num_replicas_in_sync):
            [print("{}[{}].device={}".format(it, i, example[it].values[i].device)) for it in example]
            print("label[{}].device=".format(i), label.values[i].device)
        start_time = time.time()
        sub_rng = nvtx.start_range(message="Epoch_" + str(batch+1))
        loss_val = distributed_train_step((example, label))
        nvtx.end_range(sub_rng)
        train_time += (time.time() - start_time)
        print("Step #%d\tLoss: %.6f" % (batch+1, loss_val))
    nvtx.end_range(rng)

    print("Training time = ", train_time)
