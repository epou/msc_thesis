import pandas as pd
import numpy as np
import os
import re

from src.settings import config


class IXIDataset(object):

    IMAGE_SUBJECT_COLUMN_NAME = "Subject"
    IMAGE_PATH_COLUMN_NAME = "Path"
    IMAGE_SLICE_COLUMN_NAME = "Slice"

    def __init__(self, png_dir=config["IXI Dataset"]["PNG_IMAGE_PATH"]):
        self.png_dir = png_dir
        self.df = pd.DataFrame(self._load_paths_generator())

    def _load_paths_generator(self):
        for root, _, images in os.walk(self.png_dir, topdown=False):
            for image in images:
                yield {
                    self.IMAGE_SUBJECT_COLUMN_NAME: os.path.basename(root),
                    self.IMAGE_PATH_COLUMN_NAME: os.path.join(root, image),
                    self.IMAGE_SLICE_COLUMN_NAME: int(re.search(r'\d+', image).group(0))
                }

    def filter(self, percentage, by_column=IMAGE_SUBJECT_COLUMN_NAME):
        self.df['nb_slices'] = self.df.groupby(by_column)[by_column].transform('count')
        self.df['min_slice'] = np.floor(self.df["nb_slices"] * (1 - percentage) / 2)
        self.df['max_slice'] = np.ceil(self.df["nb_slices"] * (1 + percentage) / 2)

        df_images_filtered = self.df[
            self.df[self.IMAGE_SLICE_COLUMN_NAME].between(
                self.df["min_slice"],
                self.df["max_slice"]
            )
        ]

        df_images_filtered = df_images_filtered.drop(
            columns=["nb_slices", "min_slice", "max_slice"]
        )

        self.df = df_images_filtered
