import csv
from tensorflow.keras.callbacks import Callback
import time


class ElapsedTime(Callback):

    RESULT_FILENAME = "time.csv"

    def __init__(self, result_path, logs={}):

        self.result_path = result_path

        self.global_start_time = 0
        self.global_end_time = 0

        self.epochs_times= []
        self.start_epoch_time = 0

    @property
    def global_elapsed_time(self):
        return self.end_time - self.start_time

    @property
    def mean_epoch_time(self):
        return sum(self.epochs_times)/float(len(self.epochs_times))

    def save_results(self):
        with open(self.result_path/self.RESULT_FILENAME, 'w') as csv_file:
            fieldnames = ['Elapsed_Time', 'Mean_Epoch_Time']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow(
                {
                    'Elapsed_Time': self.global_elapsed_time,
                    'Mean_Epoch_Time': self.mean_epoch_time
                }
            )

    def on_train_begin(self, logs={}):
        self.global_start_time = time.time()

    def on_train_end(self, logs={}):
        self.global_end_time = time.time()

    def on_epoch_begin(self, epoch, logs={}):
        self.start_epoch_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_times.append(
            time.time() - self.start_epoch_time
        )
