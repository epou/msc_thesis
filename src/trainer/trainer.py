from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from .callbacks import ElapsedTime
from src.utils import create_folder


class Trainer(object):

    _common_data_gen_params = dict(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rescale=1. / 255
    )

    def __init__(self, data_generator):

        self.data_generator = data_generator

    @property
    def data_generator(self):
        return self.__data_generator

    @data_generator.setter
    def data_generator(self, value):
        self.__data_generator = value

        self.train_generator = value.train_generator
        self.validation_generator = value.validation_generator
        self.test_generator = value.test_generator

    def _get_callbacks(self, results_path):

        # CSVLogger Callback
        csv_logger = CSVLogger(filename=str(results_path / "training.csv"))

        # ModelCheckpoint Callback
        model_checkpoint_dir = create_folder(name="model_checkpoints", root_path=results_path)
        checkpoint_path = model_checkpoint_dir / "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_freq='epoch',
            save_weights_only=True,
        )

        # ReduceLROnPlateau Callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            mode='min',
            verbose=1
        )

        # EarlyStopping Callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min',
            verbose=1,
            restore_best_weights=True
        )

        # Time Callback
        elapsed_time_callback = ElapsedTime(result_path=results_path)

        return [csv_logger, checkpoint, reduce_lr, early_stopping, elapsed_time_callback]

    def run(self, model, results_folder, epochs=30):
        self.train_generator.reset()
        self.validation_generator.reset()

        model_fitted = model.fit(
            self.train_generator.as_tuple,
            steps_per_epoch=self.train_generator.steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator.as_tuple,
            validation_steps=self.validation_generator.steps_per_epoch,
            callbacks=self._get_callbacks(results_path=results_folder)
        )

        return model_fitted
