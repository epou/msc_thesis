import csv
import re
from tensorflow.keras.utils import plot_model


def get_safe_filename(name):
    return re.sub(r'[^\w\d-]', '_', name)


def create_folder(name, root_path, raise_if_exist=False):
    folder = root_path / get_safe_filename(name=name)
    folder.mkdir(parents=True, exist_ok=not raise_if_exist)

    return folder


def save_model_schema_as_png(model, result_path):
    _ = plot_model(
        model=model,
        to_file=result_path/"{}_model_schema.png".format(model.name),
        show_shapes=True
    )


def save_model_params(model, result_path):
    with open(result_path / "model_params.csv", 'w') as csv_file:

        fieldnames = ['Num_params']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(
            {
                'Num_params': model.count_params(),
            }
        )

def plot_history(model_history, log_file_csv, parameters, use_only_val=False):

    nb_parameters = len(parameters) if isinstance(parameters, (list, tuple)) else 1
    if not isinstance(parameters, (list, tuple)):
        parameters = [parameters]

    df_evolution = pd.read_csv(log_file_csv, index_col="epoch")

    fig, axes = plt.subplots(nb_parameters, 1, figsize=[6.4, 4.8 * nb_parameters])

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for param, ax in zip(parameters, axes):
        df_evolution.plot(
            y=[param, "val_{}".format(param)],
            use_index=True,
            ax=ax,
            title="{} vs Epoch".format(param.title()),
            legend=False
        )
        ax.set_ylabel(param)
        ax.legend(["Train", "Valdation"], loc='center right')
    plt.show()
