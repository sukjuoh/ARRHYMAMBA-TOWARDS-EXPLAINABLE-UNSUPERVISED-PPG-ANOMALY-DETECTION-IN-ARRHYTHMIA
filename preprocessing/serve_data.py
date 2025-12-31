from torch.utils.data import DataLoader
from preprocessing.load_data import PPGTestDataset, PPG_TestSequence




def build_data_pipeline(batch_size:int,
                        dataset_importer: PPG_TestSequence,
                        kind: str,
                        window_size:int,
                        num_workers:int,
                        ) -> DataLoader:
    """
    :param kind train/valid/test
    """
    # DataLoader
    if kind == 'train':
        train_dataset = PPGTestDataset('train', dataset_importer, window_size)
        return DataLoader(train_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=False)
    elif kind == 'test':
        test_dataset = PPGTestDataset('test', dataset_importer, window_size)
        return DataLoader(test_dataset, batch_size, num_workers=num_workers, shuffle=True, drop_last=False)  # 'shuffle=True' to fairly produce the evaluation metrics on the UCR anomaly archive datasets.
    else:
        raise ValueError
