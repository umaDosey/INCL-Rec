from sklearn.model_selection import ParameterGrid
import torch
import numpy as np
from dataset import get_dataset
from utils import set_seed, init_run
from model import get_model
from trainer import get_trainer


def fitness1( aux_reg, aug_num, contrastive_reg):
    set_seed(2021)
    device = torch.device('cuda:1')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    model_config = {'name': 'DOSE_aug2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': aug_num}
    trainer_config = {'name': 'DOSEaugTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  contrastive_reg,
                      'aux_reg': aux_reg,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)

def fitness2( aux_reg, aug_num, contrastive_reg):
    set_seed(2021)
    device = torch.device('cuda:1')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    model_config = {'name': 'DOSE_aug3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': aug_num}
    trainer_config = {'name': 'DOSEaugTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  contrastive_reg,
                      'aux_reg': aux_reg,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)

def fitness3( aux_reg, aug_num, contrastive_reg):
    set_seed(2021)
    device = torch.device('cuda:1')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    model_config = {'name': 'DOSE_drop', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': aug_num}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  contrastive_reg,
                      'aux_reg': aux_reg,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def fitness4( aux_reg, aug_num, contrastive_reg):
    set_seed(2021)
    device = torch.device('cuda:1')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    model_config = {'name': 'DOSE_drop2', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': aug_num}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  contrastive_reg,
                      'aux_reg': aux_reg,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def fitness5( aux_reg, aug_num, contrastive_reg):
    set_seed(2021)
    device = torch.device('cuda:1')
    dataset_config = {'name': 'ProcessedDataset', 'path': 'data/Yelp/time',
                      'device': device}
    model_config = {'name': 'DOSE_drop3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': aug_num}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  contrastive_reg,
                      'aux_reg': aux_reg,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    return trainer.train(verbose=True)


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'aux_reg': [1.e-3], 'aug_num': [int(400000), int(600000), int(800000), int(1000000), int(1200000)], 'contrastive_reg': [1.e-2, 1.e-1, 0.5]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness1(params['aug_num'], params['contrastive_reg'], params['aux_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))

    
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'aux_reg': [1.e-3], 'aug_num': [int(400000), int(600000), int(800000), int(1000000), int(1200000)], 'contrastive_reg': [1.e-2, 1.e-1, 0.5]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness2(params['aug_num'], params['contrastive_reg'], params['aux_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))
    
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'aux_reg': [1.e-3], 'aug_num': [int(400000), int(600000), int(800000), int(1000000), int(1200000)], 'contrastive_reg': [1.e-2, 1.e-1, 0.5]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness3(params['aug_num'], params['contrastive_reg'], params['aux_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))
    
    
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'aux_reg': [1.e-3], 'aug_num': [int(400000), int(600000), int(800000), int(1000000), int(1200000)], 'contrastive_reg': [1.e-2, 1.e-1, 0.5]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness4(params['aug_num'], params['contrastive_reg'], params['aux_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))

    
    log_path = __file__[:-3]
    init_run(log_path, 2021)
    param_grid = {'aux_reg': [1.e-3], 'aug_num': [int(400000), int(600000), int(800000), int(1000000), int(1200000)], 'contrastive_reg': [1.e-2, 1.e-1, 0.5]}
    grid = ParameterGrid(param_grid)
    max_ndcg = -np.inf
    best_params = None
    for params in grid:
        ndcg = fitness5(params['aug_num'], params['contrastive_reg'], params['aux_reg'])
        print('NDCG: {:.3f}, Parameters: {:s}'.format(ndcg, str(params)))
        if ndcg > max_ndcg:
            max_ndcg = ndcg
            best_params = params
            print('Maximum NDCG!')
    print('Maximum NDCG: {:.3f}, Best Parameters: {:s}'.format(max_ndcg, str(best_params)))




if __name__ == '__main__':
    main()
