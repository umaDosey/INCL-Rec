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
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[10]
    model_config ={'name': 'DOSE_aug3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 0}
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'

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
    
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[10]
    model_config ={'name': 'DOSE_aug3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 200000}
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'

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
    
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[10]
    model_config ={'name': 'DOSE_aug3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 400000}
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'

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
    
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[10]
    model_config ={'name': 'DOSE_aug3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 600000}
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'

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
    
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[10]
    model_config ={'name': 'DOSE_aug3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 800000}
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'

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
    
    device = torch.device('cuda')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[10]
    model_config ={'name': 'DOSE_aug3', 'embedding_size': 64, 'n_layers': 3, 'device': device,
                    'dropout': 0.3, 'feature_ratio': 1, 'aug_num': 1000000}
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config)
    model = get_model(model_config, dataset)
    trainer = get_trainer(trainer_config, dataset, model)
    trainer.train(verbose=True, writer=writer)
    writer.close()


    
if __name__ == '__main__':
    main()
