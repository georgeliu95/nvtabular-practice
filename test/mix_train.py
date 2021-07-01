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
os.environ["TF_MEMORY_ALLOCATION"] = "0.6"  # fraction of free memory

import nvtabular as nvt  # noqa: E402 isort:skip
from nvtabular.framework_utils.tensorflow import layers  # noqa: E402 isort:skip
from nvtabular.loader.tensorflow import KerasSequenceLoader  # noqa: E402 isort:skip

import tensorflow as tf  # noqa: E402 isort:skip


mirrored_strategy = tf.distribute.MirroredStrategy()
# mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

class replicaWrapper(object):
    def __init__(self, replica_dataset, num_dataset=mirrored_strategy.num_replicas_in_sync):
        self.dataset = mirrored_strategy.experimental_local_results(replica_dataset)
        self.num_dataset = num_dataset

    def __iter__(self):
        return self

    def __next__(self):
        data_per_batch = list()
        for i in range(self.num_dataset):
            # print(self.dataset[i])
            next_batch = next(self.dataset[i])
            examples, labels = next_batch
            for it in labels:
                if it is not None:
                    label = it[LABEL_COLUMNS[0]][0]
            next_batch = examples, label
            data_per_batch.append(next_batch)
        return data_per_batch


def get_dataset_fn():
    idx = tf.distribute.get_replica_context().replica_id_in_sync_group
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
        global_size=mirrored_strategy.num_replicas_in_sync,
        global_rank=idx,
        seed_fn=None,
    )
    return train_dataset_tf

per_replica_dataset = mirrored_strategy.run(get_dataset_fn)

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


# with mirrored_strategy.scope():
#     def process_label_fn(inputs):
#         idx = tf.distribute.get_replica_context().replica_id_in_sync_group
#         _, tmp_labels = inputs[idx]
#         for it in tmp_labels:
#             if it is not None:
#                 labels = it[LABEL_COLUMNS[0]][0]
#         return 

#     per_replica_dataset = mirrored_strategy.run(process_label_fn, args=(mirrored_strategy.experimental_local_results(per_replica_dataset),))


print(type(per_replica_dataset))
test_replica = replicaWrapper(per_replica_dataset,)
# for it in test_replica:
#     print(it[0])
#     print(type(it), type(it[-1]))
#     break



def compute_loss(labels, predictions):
    per_example_loss = loss(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

def train_step(inputs):
    idx = int(tf.distribute.get_replica_context().replica_id_in_sync_group.device[-1])
    print(len(inputs))
    nvt_loader = inputs[idx]
    features, labels = nvt_loader

    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss 

@tf.function
def distributed_train_step(inputs):
    per_replica_losses = mirrored_strategy.run(train_step, args=(inputs,))
    return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

for batch in test_replica:
    print(len(batch))
    print(distributed_train_step(batch))

exit()



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


    # @tf.function
    def distributed_train_step(inputs):
        per_replica_losses = mirrored_strategy.run(training_step, args=(inputs,))
        # per_replica_losses = tf.distribute.get_replica_context().merge_call(training_step, args=(inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    def custom_train_step_fn(inputs, steps):
        idx = int(tf.distribute.get_replica_context().replica_id_in_sync_group.device[-1])
        nvt_loader = inputs[idx]
        print(nvt_loader, "\n"*5)


        for batch, (examples, labels) in enumerate(nvt_loader):
            if batch == steps:
                break
            loss_val = distributed_train_step((examples, labels[1][LABEL_COLUMNS[0]][0]))
            print("Step #%d\tLoss: %.6f" % (batch+1, loss_val))



    # for batch, (example, labels) in enumerate(per_replica_dataset):
    #     for it in labels:
    #         if it is not None:
    #             label = it[LABEL_COLUMNS[0]][0]
        # [print("{}.device={}".format(it, example[it].device)) for it in example]
        # print("label.device=", label.device)


mirrored_strategy.run(custom_train_step_fn, args=(mirrored_strategy.experimental_local_results(per_replica_dataset), STEPS // mirrored_strategy.num_replicas_in_sync))




