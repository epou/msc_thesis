import imageio
from imgaug import augmenters as iaa
import math
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Augmentor(object):
    def __init__(self):
        self.function = self._get_augmentor()

    def _get_augmentor(self):
        return iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.OneOf([
                        iaa.AdditiveGaussianNoise(scale=(0.04, 0.3)),
                        iaa.SaltAndPepper(0.01)
                    ]),
                    iaa.CoarseDropout(p=(0.02, 0.1)),
                    iaa.GaussianBlur(sigma=(0.1, 2.0))
                ])
            ],
            random_order=False
        )

    def process_image(self, image):
        return self.function(image=image)


class SubsetGenerator(object):
    def __init__(self, original_datagen, augmented_datagen, dataframe, x_col, target_size,
                 color_mode="grayscale", class_mode=None, **kwargs):
        self.dataframe = dataframe
        self.x_col = x_col
        self.target_size = target_size
        self.color_mode = color_mode
        self.class_mode = class_mode

        self.augmented = augmented_datagen.flow_from_dataframe(
            dataframe=self.dataframe,
            x_col=self.x_col,
            target_size=self.target_size,
            color_mode=self.color_mode,
            class_mode=self.class_mode,
            **kwargs
        )
        self.original = original_datagen.flow_from_dataframe(
            dataframe=self.dataframe,
            x_col=self.x_col,
            target_size=self.target_size,
            color_mode=self.color_mode,
            class_mode=self.class_mode,
            **kwargs
        )

    def __len__(self):
        return len(self.augmented)

    @property
    def as_tuple(self):
        return zip(self.augmented, self.original)

    @property
    def steps_per_epoch(self):
        return self.original.n // self.original.batch_size

    @property
    def image_shape(self):
        return self.augmented.image_shape

    def reset(self):
        self.augmented.reset()
        self.original.reset()


class DataGenerator(object):
    _common_data_gen_params = dict(
        samplewise_center=False,
        samplewise_std_normalization=False,
        rescale=1. / 255
    )

    def __init__(self, dataset, unique_by, path_column, image_size, augmentor=Augmentor()):
        self.dataset = dataset
        self.unique_by = unique_by
        self.image_size = image_size
        self.augmentor = augmentor

        self._augmented_datagen = ImageDataGenerator(
            **self._common_data_gen_params,
            preprocessing_function=augmentor.process_image
        )
        self._original_datagen = ImageDataGenerator(
            **self._common_data_gen_params
        )

        self.train_df, self.valid_df, self.test_df = self._split_dataset()

        self.train_generator = SubsetGenerator(
            original_datagen=self._original_datagen,
            augmented_datagen=self._augmented_datagen,
            dataframe=self.train_df,
            x_col=path_column,
            target_size=image_size,
            batch_size=16,
            seed=1
        )

        self.validation_generator = SubsetGenerator(
            original_datagen=self._original_datagen,
            augmented_datagen=self._augmented_datagen,
            dataframe=self.valid_df,
            x_col=path_column,
            target_size=image_size,
            batch_size=16,
            seed=1
        )

        self.test_generator = SubsetGenerator(
            original_datagen=self._original_datagen,
            augmented_datagen=self._augmented_datagen,
            dataframe=self.test_df,
            x_col=path_column,
            target_size=image_size,
            batch_size=1,
            suffle=False,
            seed=1
        )

    @property
    def image_shape(self):
        return self.train_generator.image_shape

    def _split_dataset(self, seed=1234, shuffle=True, split_percentages=(0.66, 0.14, 0.2)):
        if not isinstance(split_percentages, (list, tuple)):
            raise ValueError("'split' argument must be a tuple or list")
        else:
            if sum(split_percentages) != 1.0:
                raise ValueError("'split' must sum up to 1.0")

        split_train, split_val, split_test = split_percentages

        unique_choices = self.dataset.df[self.unique_by].unique().tolist()

        nb_unique = len(unique_choices)
        nb_train = math.floor(nb_unique * split_train)
        nb_val = math.floor(nb_unique * split_val)
        nb_test = nb_unique - nb_train - nb_val

        if shuffle:
            random.seed(seed)
            random.shuffle(unique_choices)

        result_choices = []
        start = 0
        for value in [nb_train, nb_val, nb_test]:
            end = start + value
            result_choices.append(
                unique_choices[start: end]
            )
            start = end

        train_choices, val_choices, test_choices = result_choices

        train_df = self.dataset.df.loc[self.dataset.df[self.unique_by].isin(train_choices)]
        val_df = self.dataset.df.loc[self.dataset.df[self.unique_by].isin(val_choices)]
        test_df = self.dataset.df.loc[self.dataset.df[self.unique_by].isin(test_choices)]

        return train_df.sample(frac=1, random_state=seed), val_df.sample(frac=1, random_state=seed), test_df.sample(
            frac=1, random_state=seed)

    def print_sample_augmented_images(self, num_augmentations=20):
        seq = get_augmentator()

        rnd_image = df_images_filtered.sample()

        image = imageio.imread(rnd_image[IMAGE_PATH_COLUMN_NAME].item())

        print("Original:")
        show_original = ia.imshow(image)

        images_aug = [seq(image=image) for _ in range(num_augmentations)]

        print("Augmented:")
        show_augmented = ia.imshow(ia.draw_grid(images_aug))

        return show_original, show_augmented