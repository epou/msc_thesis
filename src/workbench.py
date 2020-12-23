from pathlib import Path

from .utils import create_folder, save_model_schema_as_png, save_model_params

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
