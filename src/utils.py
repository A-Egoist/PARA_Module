import numpy as np
import torch
import random
import os
import yaml


def set_seed(seed=2000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_data_path(dataset):
    datasets = {'ciao': ['./data/Ciao/movie-ratings.train', './data/Ciao/movie-ratings.extend', './data/Ciao/movie-ratings.valid', './data/Ciao/movie-ratings.test'],
                'douban-book': ['./data/Douban/book/douban_book.train', './data/Douban/book/douban_book.extend', './data/Douban/book/douban_book.valid', './data/Douban/book/douban_book.test'],
                'douban-movie': ['./data/Douban/movie/douban_movie.train', './data/Douban/movie/douban_movie.extend', './data/Douban/movie/douban_movie.valid', './data/Douban/movie/douban_movie.test'],
                'ml-1m': ['./data/ml-1m/ratings.train', './data/ml-1m/ratings.extend', './data/ml-1m/ratings.valid', './data/ml-1m/ratings.test'],}
    # train, extend, valid, test
    return datasets[dataset][0], datasets[dataset][1], datasets[dataset][2], datasets[dataset][3]


def get_optimizer(backbone, method, model, lr, method_config):
    if backbone == 'MF':
        params = [
            {'params': model.user_embedding.weight, 'lr': lr}, 
            {'params': model.item_embedding.weight, 'lr': lr}
        ]
    elif backbone == 'LightGCN':
        params = [
            {'params': model.user_embedding_0.weight, 'lr': lr}, 
            {'params': model.item_embedding_0.weight, 'lr': lr}
        ]
    optimizer = torch.optim.Adam(params)
    return optimizer


def trisecting(alpha, beta, item_information):
    def classification(row):
        if row['popularity'] < alpha and row['quality'] > beta:
            return 1  # positive
        elif row['popularity'] > alpha and row['quality'] < beta:
            return 3  # negative
        else:
            return 2  # neutral
    item_information['group'] = item_information.apply(classification, axis=1)
    return item_information


def acting(item_information, gamma):
    def function(row):
        if row['group'] == 1:
            # promote
            return (row['popularity'] ** (1 / gamma)) * row['quality']
        elif row['group'] == 2:
            # maintain
            return (row['popularity']) * row['quality']
        elif row['group'] == 3:
            # suppress
            return (row['popularity'] ** gamma) * row['quality']
    item_information['coefficient'] = item_information.apply(function, axis=1)
    return item_information


def acting_without_quality(item_information, gamma):
    def function(row):
        if row['group'] == 1:
            # promote
            return (row['popularity'] ** (1 / gamma))
        elif row['group'] == 2:
            # maintain
            return (row['popularity'])
        elif row['group'] == 3:
            # suppress
            return (row['popularity'] ** gamma)
    item_information['coefficient'] = item_information.apply(function, axis=1)
    return item_information


def get_config(backbone, dataset, method):
    # args, parameters = myparser.parser()
    with open(os.path.join(os.getcwd(), 'config.yaml'), 'r', encoding='utf-8') as f:
        parameters = yaml.safe_load(f)
    # global config
    train_batch_size = parameters['batch_size']
    test_batch_size = parameters['test_batch_size']
    embedding_dim = parameters['embedding_dim']
    num_workers = parameters['num_workers']
    # dataset config
    n_layers = parameters[f'n_layers-{dataset}']
    lamb = parameters[f'lamb-{dataset}']
    num_epoch = parameters[f'{dataset}-num_epoch']
    # lr = parameters[f'{dataset}-lr']
    
    # method config
    if method == 'Base':
        lr = parameters[f'{method}-{dataset}-lr']
        method_config = None
    elif method[:4] == 'PARA':
        lr = parameters[f'PARA-{dataset}-lr']
        method_config = {'alpha': parameters[f'PARA_alpha-{dataset}'],
                         'beta': parameters[f'PARA_beta-{dataset}']}
    return num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config
