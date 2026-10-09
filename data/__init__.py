from loguru import logger

DATASET_FACTORY = {
    "VideoFilelist": "get_videofilelist_dataset",
}


def build_dataset(config, mode='train'): 
    if mode == 'train':
        dataset_name = config.DATA.train.dataset        
    elif mode == 'validate':
        dataset_name = config.DATA.validate.dataset        
    elif mode == 'test' or mode == 'predict':
        dataset_name = config.DATA.test.dataset
    else:
        raise ValueError(f'Wrong mode: {mode}. Available modes ["train", "validate", "test", "predict"]')  # to compatible with lightning datamodule stage

    if dataset_name not in DATASET_FACTORY:
        raise ValueError(f'Wrong dataset name: {dataset_name}. Available datasets: {list(DATASET_FACTORY.keys())}')
    else:
        func_name = DATASET_FACTORY[dataset_name]
        module = __import__(f"data.{dataset_name}_dataset", fromlist=[func_name])
        dataset_func = getattr(module, func_name)
        dataset = dataset_func(stage=mode, config=config)
    logger.info(f'{mode.upper()} dataset: #samples={len(dataset)}')
    return dataset
