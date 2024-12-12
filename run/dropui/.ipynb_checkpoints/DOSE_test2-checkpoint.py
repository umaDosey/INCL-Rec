from dataset import get_dataset
from model import get_model
from trainer import get_trainer
import torch
from utils import init_run
from tensorboardX import SummaryWriter
from config import get_gowalla_config, get_yelp_config, get_amazon_config
from sklearn.preprocessing import normalize


def main():
    log_path = __file__[:-3]
    init_run(log_path, 2021)


    
    device = torch.device('cuda:1')
    config = get_amazon_config(device)
    dataset_config, model_config, trainer_config = config[9]
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'
    
    model_config = {'name': 'DOSE_MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7, 'aug_rate':0.8}
    trainer_config = {'name': 'INMLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2, 'con_reg': 0.5,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
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
    data_mat = model.get_data_mat(new_dataset)[:, :dataset.n_items]
    model.normalized_data_mat = normalize(data_mat, axis=1, norm='l2')
    trainer = get_trainer(trainer_config, new_dataset, model)
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    
    device = torch.device('cuda:1')
    config = get_amazon_config(device)
    dataset_config, model_config, trainer_config = config[9]
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'
    
    model_config = {'name': 'DOSE_MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7, 'aug_rate':0.8}
    trainer_config = {'name': 'INMLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2, 'con_reg': 0.3,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
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
    data_mat = model.get_data_mat(new_dataset)[:, :dataset.n_items]
    model.normalized_data_mat = normalize(data_mat, axis=1, norm='l2')
    trainer = get_trainer(trainer_config, new_dataset, model)
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    
    
    device = torch.device('cuda:1')
    config = get_amazon_config(device)
    dataset_config, model_config, trainer_config = config[9]
    dataset_config['path'] = dataset_config['path'][:-4] + '0_dropui'
    
    model_config = {'name': 'DOSE_MultiVAE', 'layer_sizes': [64, 32],
                    'device': device, 'dropout': 0.7, 'aug_rate':0.8}
    trainer_config = {'name': 'INMLTrainer', 'optimizer': 'Adam', 'lr': 1.e-3, 'l2_reg': 1.e-5, 'kl_reg': 0.2, 'con_reg': 0.1,
                      'device': device, 'n_epochs': 1000, 'batch_size': 512, 'dataloader_num_workers': 6,
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
    data_mat = model.get_data_mat(new_dataset)[:, :dataset.n_items]
    model.normalized_data_mat = normalize(data_mat, axis=1, norm='l2')
    trainer = get_trainer(trainer_config, new_dataset, model)
    trainer.inductive_eval(dataset.n_users, dataset.n_items)
    

    model = get_model(model_config, new_dataset)
    # model.load('checkpoints/...')
    trainer = get_trainer(trainer_config, new_dataset, model)
    print('Transductive model results.')
    trainer.inductive_eval(dataset.n_users, dataset.n_items)

if __name__ == '__main__':
    main()
