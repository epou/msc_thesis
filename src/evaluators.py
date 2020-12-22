import tensorflow as tf
import pandas as pd


class BaseEvaluator(object):
    def __init__(self, subset_data_generator):
        self.subset_data_generator = subset_data_generator

    def run(self, model, results_folder):
        raise NotImplementedError


class StatisticsEvaluator(BaseEvaluator):

    def _results_generator(self, model):

        self.subset_data_generator.reset()

        augmented_generator = self.subset_data_generator.augmented
        original_generator = self.subset_data_generator.original

        for i in range(augmented_generator.n):
            pred = model(augmented_generator[i], training=False)

            pred_tensor = tf.convert_to_tensor(pred[0])
            original_tensor = tf.convert_to_tensor(original_generator[i])

            yield {
                "Filename": augmented_generator.filenames[i],
                "PSNR": tf.image.psnr(pred_tensor, original_tensor, max_val=1.0).numpy().item(),
                "SSIM": tf.image.ssim(pred_tensor, original_tensor, max_val=1.0).numpy().item(),
            }

    def run(self, model, results_folder):

        df_results = pd.DataFrame(
            self._results_generator(model=model)
        )

        df_results.to_csv(results_folder/"evaluation_metrics_raw_results.csv")

        df_results_statistics = df_results[["PSNR", "SSIM"]].describe()
        df_results_statistics.to_csv(results_folder/"evaluation_metrics_statistics_results.csv")

        print("##############################")
        print(df_results_statistics)
        print("##############################")

