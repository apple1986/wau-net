import os
import pickle

from torch.utils import data

from preprocess import UltrasoundDataSet, UltrasoundTrainInform, UltrasoundValDataSet


MAIN_DATASETS = ('BUS-BRA', 'TN3K', 'DDTI')
TEST_DATASETS = MAIN_DATASETS + ('BUS_UC',)


def _default_data_root(dataset):
    return os.path.join('./data', dataset)


def _split_file(dataset, split):
    if dataset == 'BUS_UC':
        if split != 'test':
            raise ValueError('BUS_UC is used only as the external test set.')
        return os.path.join('./datasets', 'BUS_UC_test_list.txt')
    return os.path.join('./datasets', dataset, f'{split}_list.txt')


def _load_or_collect_stats(dataset, data_root):
    # BUS_UC must use the BUS-BRA training preprocessing/statistics in the
    # cross-dataset experiment, with no BUS_UC data used for model selection.
    stats_dataset = 'BUS-BRA' if dataset == 'BUS_UC' else dataset
    stats_root = _default_data_root(stats_dataset) if dataset == 'BUS_UC' else data_root
    inform_file = os.path.join('./datasets', 'inform', f'{stats_dataset}_inform.pkl')

    if os.path.isfile(inform_file):
        with open(inform_file, 'rb') as f:
            return pickle.load(f)

    train_list = _split_file(stats_dataset, 'train')
    collector = UltrasoundTrainInform(
        data_dir=stats_root,
        classes=2,
        train_set_file=train_list,
        inform_data_file=inform_file,
    )
    return collector.collectDataAndSave()


def build_dataset_train(dataset, input_size, batch_size, train_type, random_scale,
                        random_mirror, num_workers, data_root=None,
                        train_fraction=1.0, subset_seed=1234):
    if dataset not in MAIN_DATASETS:
        raise NotImplementedError(
            f'Training supports {MAIN_DATASETS}; received {dataset!r}.'
        )

    data_root = data_root or _default_data_root(dataset)
    if train_type not in ('train', 'trainval'):
        raise ValueError("train_type must be 'train' or 'trainval'.")

    train_list = _split_file(dataset, train_type)
    val_list = _split_file(dataset, 'val')
    stats = _load_or_collect_stats(dataset, data_root)

    train_loader = data.DataLoader(
        UltrasoundDataSet(
            data_root,
            train_list,
            crop_size=input_size,
            scale=random_scale,
            mirror=random_mirror,
            mean=stats['mean'],
            train_fraction=train_fraction,
            subset_seed=subset_seed,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = data.DataLoader(
        UltrasoundValDataSet(
            data_root,
            val_list,
            image_size=input_size,
            mean=stats['mean'],
        ),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return stats, train_loader, val_loader


def build_dataset_test(dataset, num_workers, input_size=(256, 256), data_root=None):
    if dataset not in TEST_DATASETS:
        raise NotImplementedError(
            f'Testing supports {TEST_DATASETS}; received {dataset!r}.'
        )

    data_root = data_root or _default_data_root(dataset)
    test_list = _split_file(dataset, 'test')
    stats = _load_or_collect_stats(dataset, data_root)

    test_loader = data.DataLoader(
        UltrasoundValDataSet(
            data_root,
            test_list,
            image_size=input_size,
            mean=stats['mean'],
        ),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return stats, test_loader
