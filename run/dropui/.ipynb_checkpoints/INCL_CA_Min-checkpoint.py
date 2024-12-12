from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)

    device = torch.device('cuda')
    config = get_yelp_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'
    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 0.7, 'aug_num': 400000, 'aug_rate':0.8}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  0.3,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    dataset_config['path'] = dataset_config['path'][:-7]
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    model.norm_adj = model.generate_graph(new_dataset)
    model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
    model.update_feat_mat()
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Inductive results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

    
    device = torch.device('cuda')
    config = get_yelp_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + '1_dropui'
    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 0.7, 'aug_num': 400000, 'aug_rate':0.8}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  0.3,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    dataset_config['path'] = dataset_config['path'][:-7]
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    model.norm_adj = model.generate_graph(new_dataset)
    model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
    model.update_feat_mat()
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Inductive results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    device = torch.device('cuda')
    
    config = get_yelp_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + '2_dropui'
    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 0.7, 'aug_num': 400000, 'aug_rate':0.8}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  0.3,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    dataset_config['path'] = dataset_config['path'][:-7]
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    model.norm_adj = model.generate_graph(new_dataset)
    model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
    model.update_feat_mat()
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Inductive results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

    device = torch.device('cuda')
    config = get_yelp_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + '3_dropui'
    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 0.7, 'aug_num': 400000, 'aug_rate':0.8}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  0.3,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    dataset_config['path'] = dataset_config['path'][:-7]
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    model.norm_adj = model.generate_graph(new_dataset)
    model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
    model.update_feat_mat()
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Inductive results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    
    device = torch.device('cuda')
    config = get_yelp_config(device)
    dataset_config, model_config, trainer_config = config[2]
    dataset_config['path'] = dataset_config['path'][:-4] + '4_dropui'
    model_config = {'name': 'DOSE_aug', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 0.7, 'aug_num': 400000, 'aug_rate':0.8}
    trainer_config = {'name': 'DOSEdropTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 0., 'contrastive_reg':  0.3,
                      'aux_reg': 0.001,
                      'device': device, 'n_epochs': 1000, 'batch_size': 2048, 'dataloader_num_workers': 6,
                      'test_batch_size': 512, 'topks': [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]}

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()

    dataset_config['path'] = dataset_config['path'][:-7]
    new_dataset = get_dataset(dataset_config)
    model.config['dataset'] = new_dataset
    model.n_users, model.n_items = new_dataset.n_users, new_dataset.n_items
    model.norm_adj = model.generate_graph(new_dataset)
    model.feat_mat, _, _, model.row_sum = model.generate_feat(new_dataset, is_updating=True)
    model.update_feat_mat()
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Inductive results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    
if __name__ == '__main__':
    main()
