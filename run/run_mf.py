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

    device = torch.device('cuda:1')
    config = get_gowalla_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + str(1)

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config) # datasetのクラス
    model = get_model(model_config, dataset) # modelのクラウス
    trainer = get_trainer(trainer_config, dataset, model) #trainerのクラス
    trainer.train(verbose=True, writer=writer)
    writer.close()
    results, _, _ = trainer.eval('test')
    print('Test result. {:s}'.format(results))
    
    device = torch.device('cuda:1')
    config = get_yelp_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + str(1)

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config) # datasetのクラス
    model = get_model(model_config, dataset) # modelのクラウス
    trainer = get_trainer(trainer_config, dataset, model) #trainerのクラス
    trainer.train(verbose=True, writer=writer)
    writer.close()
    results, _, _ = trainer.eval('test')
    print('Test result. {:s}'.format(results))
    
    device = torch.device('cuda:1')
    config = get_amazon_config(device)
    dataset_config, model_config, trainer_config = config[0]
    dataset_config['path'] = dataset_config['path'][:-4] + str(1)

    writer = SummaryWriter(log_path)
    dataset = get_dataset(dataset_config) # datasetのクラス
    model = get_model(model_config, dataset) # modelのクラウス
    trainer = get_trainer(trainer_config, dataset, model) #trainerのクラス
    trainer.train(verbose=True, writer=writer)
    writer.close()
    results, _, _ = trainer.eval('test')
    print('Test result. {:s}'.format(results))


if __name__ == '__main__':
    main()
