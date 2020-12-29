from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from random import randint

from .utils import create_folder, save_model_schema_as_png, save_model_params, get_random_images

from src.settings import config


class Workbench(object):
    def __init__(self, data_generator, trainer, evaluators, name, results_path=config["Common"]["results_dir"]):
        self.data_generator = data_generator
        self.trainer = trainer(data_generator=data_generator)
        self.evaluators = [x(subset_data_generator=data_generator.test_generator) for x in evaluators]
        self.name = "{}_workbench".format(name)

        self.results_path = create_folder(
            name=self.name,
            root_path=Path(results_path),
            raise_if_exist=False
        )

    @property
    def image_shape(self):
        return self.data_generator.image_shape

    def run(self, model, *args, **kwargs):

        model_results_folder = create_folder(
            name=model.name,
            root_path=self.results_path,
            raise_if_exist=True
        )

        save_model_schema_as_png(model=model, result_path=model_results_folder)
        save_model_params(model=model, result_path=model_results_folder)

        model_fitted = self.trainer.run(
            model=model,
            results_folder=model_results_folder,
            *args, **kwargs
        )

        for evaluator in self.evaluators:
            evaluator.run(model=model, results_folder=model_results_folder)

        return model_fitted

    def compare_model_predictions(self, models, number_images=5, title=None, figsize=(14, 19), images=None):

        images = get_random_images(
            generator=self.data_generator.test_generator,
            number=number_images,
            reset=True
        ) if not images else images

        number_images = len(images) if images else number_images

        fig, axes = plt.subplots(
            nrows=(2 + len(models)),
            ncols=number_images,
            figsize=figsize
        )

        if title:
            fig.suptitle(title, fontsize=16)

        row_names = [
            "Input",
            "Original"
        ]
        for model in models:
            row_names.append(model.name)

        for ax, row in zip(axes[:, 0], row_names):
            ax.set_ylabel(row, size='x-large')

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

        for ax, image in zip(axes.transpose(), images):

            augmented_image = image[0]
            original_image = image[1]

            ax[0].imshow(augmented_image, cmap="gray")
            ax[1].imshow(original_image, cmap="gray")

            for model, ax in zip(models, ax[2:]):
                ax.imshow(
                    model(
                        augmented_image[np.newaxis, ..., np.newaxis],
                        training=False
                    )[0, :, :, 0],
                    cmap="gray"
                )

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.align_labels()
        fig.tight_layout()
        return fig
