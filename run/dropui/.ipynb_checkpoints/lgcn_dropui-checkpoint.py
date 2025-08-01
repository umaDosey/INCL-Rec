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
    dataset_config, model_config, trainer_config = config[1]
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
    with torch.no_grad():
        old_embedding = model.embedding.weight
        model.embedding = torch.nn.Embedding(new_dataset.n_users + new_dataset.n_items, model.embedding_size, device=device)
        model.embedding.weight[:, :] = old_embedding.mean(dim=0)[None, :].expand(model.embedding.weight.shape)
        model.embedding.weight[:dataset.n_users, :] = old_embedding[:dataset.n_users, :]
        model.embedding.weight[new_dataset.n_users:new_dataset.n_users + dataset.n_items, :] = \
            old_embedding[dataset.n_users:, :]
    trainer = get_trainer(trainer_config, new_dataset, model)
    trainer.inductive_eval(dataset.n_users, dataset.n_items)


if __name__ == '__main__':
    main()
