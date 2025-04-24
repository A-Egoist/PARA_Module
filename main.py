import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from collections import defaultdict
from tqdm import tqdm
import time

from src.utils import *
from src.metrics import *
from src import data_processing
from src.models import *


def grid_search(random_seed, device, backbone, method, dataset, mode, args):
    set_seed(random_seed)
    log_file = logger.add(f'./logs/grid_search/{dataset}-{backbone}-{method}-{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.log', encoding='utf-8')
    logger.info(f'backbone: {backbone}, device: {device}, dataset: {dataset}, method: {method}, patience: {args.patience}, eval_interval: {args.eval_interval}')
    
    # config
    num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config = get_config(backbone, dataset, method)
    logger.info(f"random_seed={random_seed}, num_epoch={num_epoch}, batch_size={train_batch_size}, lr={lr}, embedding_dim={embedding_dim}, n_layers={n_layers}, lamb={lamb}, method_config={method_config}")

    # data processing 
    train_data_path, extend_data_path, valid_data_path, test_data_path = get_data_path(dataset)
    num_users, num_items, train_data, valid_data, test_data, item_information = data_processing.load_data(dataset, train_data_path, valid_data_path, test_data_path)

    graph = None
    if backbone == 'LightGCN':
        graph = data_processing.get_graph(dataset, num_users, num_items, train_data).to(device)
    
    alpha_list = [np.round(i, 2) for i in np.linspace(0, 1, 11)]
    beta_list = [np.round(i, 2) for i in np.linspace(0, 1, 11)]
    best_NDCG_20 = 0
    for alpha in alpha_list:
        for beta in beta_list:
            item_information = trisecting(alpha, beta, item_information)
            item_information = acting(item_information, args.gamma)
            item_adjustment_coeeficient = torch.tensor(item_information['coefficient'].values, dtype=torch.float32, device=device)

            model = PARA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_adjustment_coeeficient, method_config, graph)
            
            model.load_state_dict(torch.load(f'./save_model/{backbone}-{method}-{dataset}_final.pt'))
            model.to(device)

            test_metrics_dict = model_test(model, test_data, test_batch_size, train_data, item_information)
            if test_metrics_dict['NDCG@20'] > best_NDCG_20:
                best_NDCG_20 = test_metrics_dict['NDCG@20']
            logger.info('----------------------------------------------------')
            logger.info(f'alpha={alpha}, beta={beta}:')
            logger.info(test_metrics_dict)
            logger.info('----------------------------------------------------')
    logger.info(f'Best NDCG@20: {best_NDCG_20}')
    logger.remove(log_file)


def model_valid(model, valid_data, eval_batch_size, train_data, k_values=[5, 10, 20]):
    model.eval()
    metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': []}
    valid_users = valid_data['user'].unique()  # the set composed of all users in test data, ndarray
    num_user_batchs = len(valid_users) // eval_batch_size + 1
    train_user_items = train_data.groupby('user')['item'].apply(list).to_dict()  # the set composed of $\mathbb{I}_{u}$ where $u\in\mathbb{U}$
    items_mask = []  # 2d
    for user in valid_users:
        if user in train_user_items.keys():
            items_mask.append(train_user_items[user])
        else:
            items_mask.append([])  # ensure len(items_removie) == len(test_users)
    with torch.no_grad():
        for batch_id in tqdm(range(num_user_batchs)):
            user_batch = valid_users[batch_id * eval_batch_size: (batch_id + 1) * eval_batch_size]  # get a batch of users
            items_remove_batch = items_mask[batch_id * eval_batch_size: (batch_id + 1) * eval_batch_size]
            user_ids = torch.from_numpy(user_batch).long().to(device)

            # get prediction value
            prediction_batch = model.predict(user_ids).detach().cpu()
            
            # remove items that user have interacted in train data
            for index in range(len(items_remove_batch)):
                prediction_batch[index][np.array(items_remove_batch[index])] = -1

            # get top-k recommendation lists
            _, top_k_indices_sorted = torch.topk(prediction_batch, k=k_values[-1], dim=1)
            top_k_indices_sorted = top_k_indices_sorted.numpy()

            # get true list
            ground_truth = []
            for user in user_batch:
                ground_truth.append(valid_data.loc[valid_data['user'] == user, 'item'].values.reshape(-1))

            # compute performances
            for t, r in zip(ground_truth, top_k_indices_sorted):
                for top_k in k_values:
                    metrics_dict[f'Recall@{top_k}'].append(get_Recall(t, r[:top_k]))
                    metrics_dict[f'NDCG@{top_k}'].append(get_NDCG(t, r[:top_k], top_k))
    
    metrics_dict_mean = {}
    for key, value in metrics_dict.items():
        value_mean = np.round(np.mean(value), 4)
        metrics_dict_mean[key] = value_mean
    return metrics_dict_mean


def model_test(model, test_data, test_batch_size, train_data, item_information, k_values=[5, 10, 20]):
    model.eval()
    test_metrics_dict = {'Recall@5': [], 'NDCG@5': [], 'Recall@10': [], 'NDCG@10': [], 'Recall@20': [], 'NDCG@20': []}
    test_users = test_data['user'].unique()  # the set composed of all users in test data, ndarray
    num_user_batchs = len(test_users) // test_batch_size + 1
    train_user_items = train_data.groupby('user')['item'].apply(list).to_dict()  # the set composed of $\mathbb{I}_{u}$ where $u\in\mathbb{U}$
    items_mask = []  # 2d
    for user in test_users:
        if user in train_user_items.keys():
            items_mask.append(train_user_items[user])
        else:
            items_mask.append([])  # ensure len(items_removie) == len(test_users)
    with torch.no_grad():
        for batch_id in tqdm(range(num_user_batchs)):
            user_batch = test_users[batch_id * test_batch_size: (batch_id + 1) * test_batch_size]  # get a batch of users
            items_remove_batch = items_mask[batch_id * test_batch_size: (batch_id + 1) * test_batch_size]
            user_ids = torch.from_numpy(user_batch).long().to(device)

            # get prediction value
            prediction_batch = model.predict(user_ids).detach().cpu()
            
            # remove items that user have interacted in train data
            for index in range(len(items_remove_batch)):
                prediction_batch[index][np.array(items_remove_batch[index])] = -1

            # get top-k recommendation lists
            _, top_k_indices_sorted = torch.topk(prediction_batch, k=k_values[-1], dim=1)
            top_k_indices_sorted = top_k_indices_sorted.numpy()

            # get true list
            ground_truth = []
            for user in user_batch:
                ground_truth.append(test_data.loc[test_data['user'] == user, 'item'].values.reshape(-1))

            # compute performances
            for t, r in zip(ground_truth, top_k_indices_sorted):
                for top_k in k_values:
                    test_metrics_dict[f'Recall@{top_k}'].append(get_Recall(t, r[:top_k]))
                    test_metrics_dict[f'NDCG@{top_k}'].append(get_NDCG(t, r[:top_k], top_k))

    test_metrics_dict_mean = {}
    for key, value in test_metrics_dict.items():
        value_mean = np.round(np.mean(value), 4)
        test_metrics_dict_mean[key] = value_mean

    return test_metrics_dict_mean


def run(random_seed, device, backbone, method, dataset, mode, args):
    set_seed(random_seed)
    log_file = logger.add(f'./logs/{dataset}/{backbone}-{method}-{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}.log', encoding='utf-8')
    logger.info(f'backbone: {backbone}, device: {device}, dataset: {dataset}, method: {method}, patience: {args.patience}, eval_interval: {args.eval_interval}')
    
    # config
    num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config = get_config(backbone, dataset, method)
    logger.info(f"random_seed={random_seed}, num_epoch={num_epoch}, batch_size={train_batch_size}, lr={lr}, embedding_dim={embedding_dim}, n_layers={n_layers}, lamb={lamb}, method_config={method_config}")

    # data processing 
    train_data_path, extend_data_path, valid_data_path, test_data_path = get_data_path(dataset)
    num_users, num_items, train_data, valid_data, test_data, item_information = data_processing.load_data(dataset, train_data_path, valid_data_path, test_data_path)

    graph = None
    if backbone == 'LightGCN':
        graph = data_processing.get_graph(dataset, num_users, num_items, train_data).to(device)
    
    train_dataset = data_processing.BPRDataset(train_data[['user', 'item']], num_items, 4, True)
    train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)

    if method == 'PARA':
        item_information = trisecting(method_config['alpha'], method_config['beta'], item_information)
        item_information = acting(item_information, args.gamma)
        item_adjustment_coeeficient = torch.tensor(item_information['coefficient'].values, dtype=torch.float32, device=device)

    if method == 'Base':
        model = Base(backbone, device, num_users, num_items, embedding_dim, n_layers, graph)
    elif method == 'PARA':
        model = PARA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_adjustment_coeeficient, method_config, graph)
    
    if mode == 'test' or args.pre_trained:
        model.load_state_dict(torch.load(f'./save_model/{backbone}-{method}-{dataset}_final.pt'))
    model.to(device)

    if mode == 'train' or mode == 'both':
        # optimizer
        optimizer = get_optimizer(backbone, method, model, lr, method_config)

        # train
        model.train()
        train_loader.dataset.negative_sample(dataset, extend_data_path)  # generate negative samples
        loss_record = []
        best_metrics_dict = {'Best_Recall@5': -1, 'Best_NDCG@5': -1, 'Best_Recall@10': -1, 'Best_NDCG@10': -1, 'Best_Recall@20': -1, 'Best_NDCG@20': -1}
        best_epoch = {'Best_epoch_Recall@5': -1, 'Best_epoch_NDCG@5': -1, 'Best_epoch_Recall@10': -1, 'Best_epoch_NDCG@10': -1, 'Best_epoch_Recall@20': -1, 'Best_epoch_NDCG@20': -1}
        bad_count = 0  # patience
        for epoch in range(num_epoch):
            model.train()
            flag_update_metric = 0  # used for validation
            loss_sum = torch.tensor([0], dtype=torch.float32).to(device)
            for user_ids, item_i_ids, item_j_ids in tqdm(train_loader):
                user_ids = user_ids.to(device)
                item_i_ids = item_i_ids.to(device)
                item_j_ids = item_j_ids.to(device)
                if method == 'Base':
                    prediction_i, prediction_j, reg_loss = model(user_ids, item_i_ids, item_j_ids)
                    loss = -1 * (prediction_i - prediction_j).sigmoid().log().sum() + lamb * reg_loss
                elif method == 'PARA':
                    prediction_i, prediction_j, reg_loss = model(user_ids, item_i_ids, item_j_ids)
                    loss = -1 * (prediction_i - prediction_j).sigmoid().log().sum() + lamb * reg_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            
            logger.info(f'{backbone}-{method} on {dataset}: Epoch [{epoch + 1}/{num_epoch}], Loss={np.round(loss_sum.item(), 4)}')
            loss_record.append(loss_sum.item())

            # validation
            if epoch != 0 and epoch % args.eval_interval == 0:
                metrics_dict = model_valid(model, valid_data, test_batch_size, train_data)

                for key_temp, values_temp in metrics_dict.items():
                    if values_temp > best_metrics_dict['Best_' + key_temp]:
                        flag_update_metric = 1
                        bad_count = 0
                        best_metrics_dict['Best_' + key_temp] = values_temp
                        best_epoch['Best_epoch_' + key_temp] = epoch
                        if not os.path.exists(args.save_path):
                            os.makedirs(args.save_path)
                        torch.save(model.state_dict(), f'./save_model/{backbone}-{method}-{dataset}_temp.pt')
                
                if flag_update_metric == 0:
                    bad_count += 1
                else:
                    logger.info('-------------Temporary Best Validation--------------')
                    logger.info(best_metrics_dict)
                    logger.info(best_epoch)
                    logger.info('----------------------------------------------------')
                if bad_count >= args.patience:
                    if not os.path.exists(args.save_path):
                        os.makedirs(args.save_path)
                    torch.save(model.state_dict(), f'./save_model/{backbone}-{method}-{dataset}_final.pt')
                    logger.info('Early stopped.')
                    break
        logger.info('---------------Final Best Validation----------------')
        logger.info(best_metrics_dict)
        logger.info(best_epoch)
        logger.info('----------------------------------------------------')

    if mode == 'test' or mode == 'both':
        # test
        test_metrics_dict = model_test(model, test_data, test_batch_size, train_data, item_information)
        logger.info('----------------------Testing-----------------------')
        logger.info(test_metrics_dict)
        logger.info('----------------------------------------------------')
        
    logger.remove(log_file)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--backbone', type=str, default='MF', choices=['MF', 'LightGCN'])
    parser.add_argument('--method', type=str, default='Base', choices=['Base', 'PARA'])
    parser.add_argument('--dataset', type=str, default='ciao', choices=['ciao', 'douban-book', 'douban-movie', 'ml-1m'])
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'test', 'both', 'grid_search'])
    parser.add_argument('--random_seed', type=int, default=2000, help="random seed")
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop')
    parser.add_argument('--eval_interval', type=int, default=2, help='the number of epoch to eval')
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--save_path', type=str, default=os.path.join(os.getcwd(), 'save_model'), help='save model path')
    parser.add_argument('--pre_trained', type=bool, default=False)
    args = parser.parse_args()

    if args.mode == 'grid_search':
        grid_search(args.random_seed, device, args.backbone, args.method, args.dataset, args.mode, args)
    else:
        run(args.random_seed, device, args.backbone, args.method, args.dataset, args.mode, args)
