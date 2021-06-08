import numpy as np
import pandas as pd
import nvtabular as nvt
import os
import shutil
import glob

STEPS = 100
GLOBAL_BATCH_SIZE = 2<<15
EMBEDDING_SIZE = 512
# FEATURE_SIZE = 8*1024
NUM_FEATURES = 3
FEATURE_COLUMNS = ["features_" + str(idx) for idx in range(NUM_FEATURES)]
LABEL_COLUMNS = ["labels"]

if __name__ == "__main__":
    # features_array = np.random.normal(size=(GLOBAL_BATCH_SIZE, FEATURE_SIZE)).astype(np.float32)
    features_array = np.arange(start=0, stop=GLOBAL_BATCH_SIZE * STEPS, dtype=np.int64)
    labels_array = np.random.uniform(low=0, high=2, size=(GLOBAL_BATCH_SIZE * STEPS, 1)).astype(np.int64)

    data_dict = {LABEL_COLUMNS[0]:labels_array.tolist()}
    data_dict.update({it: features_array.tolist() for it in FEATURE_COLUMNS})

    df = pd.DataFrame(data=data_dict, columns=(FEATURE_COLUMNS + LABEL_COLUMNS))

    print(df.head(5))

    df.to_parquet("./data/train.parquet")

    train_dataset = nvt.Dataset(df)

    workflow = nvt.Workflow((nvt.ColumnGroup(FEATURE_COLUMNS) >> nvt.ops.Categorify()) + nvt.ColumnGroup(LABEL_COLUMNS))

    print(workflow.fit(train_dataset))

    # dict_dtypes = {}
    # for col in FEATURE_COLUMNS:
    #     dict_dtypes[col] = np.int64
    # for col in LABEL_COLUMNS:
    #     dict_dtypes[col] = np.float32


    if os.path.exists("./data/convert"):
        shutil.rmtree("./data/convert")
    out_files_per_proc = 4  # Number of output files per worker
    workflow.transform(train_dataset).to_parquet(
        output_path="./data/convert",
        shuffle=None,
        out_files_per_proc=out_files_per_proc,
        # dtypes=dict_dtypes,
    )

    print(nvt.ops.get_embedding_sizes(workflow))

    # new_df = pd.concat([pd.read_parquet(it) for it in sorted(glob.glob("./data/convert/*.parquet"))])
    # new_df.to_parquet("./data/convert/tf_train.parquet")
