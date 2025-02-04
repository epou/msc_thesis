from pathlib import Path
import tensorflow as tf
import os
import pandas as pd
import re

from src.evaluators import StatisticsEvaluator
from src.data.datasets import IXIDataset
from src.data.generator import DataGenerator
from src.trainer import Trainer
from src.workbench import Workbench


def use_gpu(enable=True):
    if enable:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        tf.config.set_visible_devices([], 'GPU')


def load_experiment_workbench(name):

    dataset = IXIDataset()
    dataset.filter(percentage=0.45)
    data_generator = DataGenerator(
        dataset=dataset,
        unique_by=dataset.IMAGE_SUBJECT_COLUMN_NAME,
        path_column=dataset.IMAGE_PATH_COLUMN_NAME,
        image_size=(224, 224)
    )

    return Workbench(
        data_generator=data_generator,
        trainer=Trainer,
        evaluators=[StatisticsEvaluator],
        name=name
    )


def get_best_weights(results_path):
    df = pd.read_csv(results_path/"training.csv", index_col="epoch")

    best_epoch = df["val_loss"].idxmin() + 1

    best_weights_file = None

    model_checkpoint_path = Path(results_path/"model_checkpoints")

    for weight_file in os.listdir(model_checkpoint_path):
        if int(re.search(r'\d+', weight_file).group(0)) == best_epoch:
            best_weights_file = model_checkpoint_path/weight_file

    return best_weights_file
