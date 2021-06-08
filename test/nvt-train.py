import glob
import os
import cupy
import numpy as np
import time

import nvtx
from create_data import STEPS, GLOBAL_BATCH_SIZE, EMBEDDING_SIZE, FEATURE_COLUMNS, LABEL_COLUMNS

# we can control how much memory to give tensorflow with this environment variable
# IMPORTANT: make sure you do this before you initialize TF's runtime, otherwise
# TF will have claimed all free GPU memory
os.environ["TF_MEMORY_ALLOCATION"] = "0.3"  # fraction of free memory

import nvtabular as nvt  # noqa: E402 isort:skip
from nvtabular.framework_utils.tensorflow import layers  # noqa: E402 isort:skip
from nvtabular.loader.tensorflow import KerasSequenceLoader  # noqa: E402 isort:skip

import tensorflow as tf  # noqa: E402 isort:skip
import horovod.tensorflow as hvd  # noqa: E402 isort:skip


hvd.init()


train_dataset_tf = KerasSequenceLoader(
    sorted(glob.glob("./data/convert/*.parquet")),  # you could also use a glob pattern
    batch_size=GLOBAL_BATCH_SIZE,
    label_names=LABEL_COLUMNS,
    cat_names=FEATURE_COLUMNS,
    cont_names=None,
    engine="parquet",
    shuffle=False,
    buffer_size=0.06,  # how many batches to load at once
    parts_per_chunk=1,
    global_size=hvd.size(),
    global_rank=hvd.rank(),
    seed_fn=None,
)


inputs = {}  # tf.keras.Input placeholders for each feature to be used
emb_layers = []  # output of all embedding layers, which will be concatenated
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
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True,  reduction=tf.keras.losses.Reduction.NONE)
# loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

def compute_loss(labels, predictions):
    per_example_loss = loss(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

@tf.function(experimental_relax_shapes=True)
def training_step(examples, labels, first_batch):
    with tf.GradientTape() as tape:
        probs = model(examples, training=True)
        loss_value = compute_loss(labels, probs)
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


train_time = 0
rng = nvtx.start_range(message="Training phase")
# Horovod: adjust number of steps based on number of GPUs.
for batch, (example, labels) in enumerate(train_dataset_tf):
    for it in labels:
        if it is not None:
            label = it[LABEL_COLUMNS[0]][0]
    start_time = time.time()
    sub_rng = nvtx.start_range(message="Epoch_" + str(batch+1))
    loss_value = training_step(example, label, batch == 0)
    nvtx.end_range(sub_rng)
    train_time += (time.time() - start_time)
    if hvd.local_rank() == 0:
        # print("label.device=", label.device)
        print("Step #%d\tLoss: %.6f" % (batch+1, loss_value))

hvd.join()
nvtx.end_range(rng)

if hvd.local_rank() == 0:
    print("Training time = ", train_time)

# Horovod: save checkpoints only on worker 0 to prevent other workers from
# corrupting it.
# if hvd.rank() == 0:
#     checkpoint.save(checkpoint_dir)

