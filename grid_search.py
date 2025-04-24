import numpy as np
import pandas as pd
import torch
from loguru import logger
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils import *
from src.metrics import *
from src import data_processing
from src.models import *


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


def grid_search_alpha(device, backbone, method, dataset, beta, gamma):
    num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config = get_config(backbone, dataset, method)
    train_data_path, extend_data_path, valid_data_path, test_data_path = get_data_path(dataset)
    num_users, num_items, train_data, valid_data, test_data, item_information = data_processing.load_data(dataset, train_data_path, valid_data_path, test_data_path)

    graph = None
    if backbone == 'LightGCN':
        graph = data_processing.get_graph(dataset, num_users, num_items, train_data).to(device)
    
    alpha_list = [np.round(i, 2) for i in np.linspace(0, 1, 11)]
    Recall_at_20_list = []
    NDCG_at_20_list = []
    for alpha in alpha_list:
        item_information = trisecting(alpha, beta, item_information)
        item_information = acting(item_information, gamma)
        item_adjustment_coeeficient = torch.tensor(item_information['coefficient'].values, dtype=torch.float32, device=device)

        model = PARA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_adjustment_coeeficient, method_config, graph)
        
        model.load_state_dict(torch.load(f'./save_model/{backbone}-{method}-{dataset}_final.pt'))
        model.to(device)

        test_metrics_dict = model_test(model, test_data, test_batch_size, train_data, item_information)
        Recall_at_20_list.append(test_metrics_dict['Recall@20'])
        NDCG_at_20_list.append(test_metrics_dict['NDCG@20'])
    
    plt.figure(figsize=(4, 3))
    
    plt.plot(alpha_list, Recall_at_20_list, marker='o', linewidth=2, markersize=6, label='Recall@20', color='#007acc')
    plt.plot(alpha_list, NDCG_at_20_list, marker='s', linewidth=2, markersize=6, label='NDCG@20', color='#ff8c00')
    
    plt.xlabel(r'$\alpha$', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xticks(alpha_list)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    # plt.savefig(f'output/grid_search/{dataset}/{backbone}_alpha_{beta}_{gamma}.png', dpi=300)
    plt.savefig(f'output/grid_search/{dataset}/{backbone}_alpha_{beta}_{gamma}.pdf')
    # plt.show()


def grid_search_beta(device, backbone, method, dataset, alpha, gamma):
    num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config = get_config(backbone, dataset, method)
    train_data_path, extend_data_path, valid_data_path, test_data_path = get_data_path(dataset)
    num_users, num_items, train_data, valid_data, test_data, item_information = data_processing.load_data(dataset, train_data_path, valid_data_path, test_data_path)

    graph = None
    if backbone == 'LightGCN':
        graph = data_processing.get_graph(dataset, num_users, num_items, train_data).to(device)
    
    beta_list = [np.round(i, 2) for i in np.linspace(0, 1, 11)]
    Recall_at_20_list = []
    NDCG_at_20_list = []
    for beta in beta_list:
        item_information = trisecting(alpha, beta, item_information)
        item_information = acting(item_information, gamma)
        item_adjustment_coeeficient = torch.tensor(item_information['coefficient'].values, dtype=torch.float32, device=device)

        model = PARA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_adjustment_coeeficient, method_config, graph)
        
        model.load_state_dict(torch.load(f'./save_model/{backbone}-{method}-{dataset}_final.pt'))
        model.to(device)

        test_metrics_dict = model_test(model, test_data, test_batch_size, train_data, item_information)
        Recall_at_20_list.append(test_metrics_dict['Recall@20'])
        NDCG_at_20_list.append(test_metrics_dict['NDCG@20'])
    
    plt.figure(figsize=(4, 3))
    
    plt.plot(beta_list, Recall_at_20_list, marker='o', linewidth=2, markersize=6, label='Recall@20', color='#007acc')
    plt.plot(beta_list, NDCG_at_20_list, marker='s', linewidth=2, markersize=6, label='NDCG@20', color='#ff8c00')
    
    plt.xlabel(r'$\beta$', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xticks(beta_list)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    # plt.savefig(f'output/grid_search/{dataset}/{backbone}_beta_{alpha}_{gamma}.png', dpi=300)
    plt.savefig(f'output/grid_search/{dataset}/{backbone}_beta_{alpha}_{gamma}.pdf')
    # plt.show()


def grid_search_gamma(device, backbone, method, dataset, alpha, beta):
    num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config = get_config(backbone, dataset, method)
    train_data_path, extend_data_path, valid_data_path, test_data_path = get_data_path(dataset)
    num_users, num_items, train_data, valid_data, test_data, item_information = data_processing.load_data(dataset, train_data_path, valid_data_path, test_data_path)

    graph = None
    if backbone == 'LightGCN':
        graph = data_processing.get_graph(dataset, num_users, num_items, train_data).to(device)
    
    gamma_list = [np.round(i, 0) for i in np.linspace(1, 10, 10)]
    Recall_at_20_list = []
    NDCG_at_20_list = []
    for gamma in gamma_list:
        item_information = trisecting(alpha, beta, item_information)
        item_information = acting(item_information, gamma)
        item_adjustment_coeeficient = torch.tensor(item_information['coefficient'].values, dtype=torch.float32, device=device)

        model = PARA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_adjustment_coeeficient, method_config, graph)
        
        model.load_state_dict(torch.load(f'./save_model/{backbone}-{method}-{dataset}_final.pt'))
        model.to(device)

        test_metrics_dict = model_test(model, test_data, test_batch_size, train_data, item_information)
        Recall_at_20_list.append(test_metrics_dict['Recall@20'])
        NDCG_at_20_list.append(test_metrics_dict['NDCG@20'])
    
    plt.figure(figsize=(4, 3))
    
    plt.plot(gamma_list, Recall_at_20_list, marker='o', linewidth=2, markersize=6, label='Recall@20', color='#007acc')
    plt.plot(gamma_list, NDCG_at_20_list, marker='s', linewidth=2, markersize=6, label='NDCG@20', color='#ff8c00')
    
    plt.xlabel(r'$\gamma$', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xticks(gamma_list)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    # plt.savefig(f'output/grid_search/{dataset}/{backbone}_gamma_{alpha}_{beta}.png', dpi=300)
    plt.savefig(f'output/grid_search/{dataset}/{backbone}_gamma_{alpha}_{beta}.pdf')
    # plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--backbone', type=str, default='MF', choices=['MF', 'LightGCN'])
    parser.add_argument('--method', type=str, default='base', choices=['base', 'PARA'])
    parser.add_argument('--dataset', type=str, default='ciao', choices=['ciao', 'douban-book', 'douban-movie', 'ml-1m'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'both', 'grid_search'])
    parser.add_argument('--random_seed', type=int, default=2000, help="random seed")
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop')
    parser.add_argument('--eval_interval', type=int, default=2, help='the number of epoch to eval')
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--save_path', type=str, default=os.path.join(os.getcwd(), 'save_model'), help='save model path')
    parser.add_argument('--pre_trained', type=bool, default=False)
    args = parser.parse_args()
    set_seed(args.random_seed)

    if args.dataset == 'ciao':
        grid_search_alpha(device, args.backbone, args.method, args.dataset, beta=0.8, gamma=4)
        grid_search_beta(device, args.backbone, args.method, args.dataset, alpha=0.8, gamma=4)
        grid_search_gamma(device, args.backbone, args.method, args.dataset, alpha=0.8, beta=0.8)
    elif args.dataset == 'douban-book':
        grid_search_alpha(device, args.backbone, args.method, args.dataset, beta=0.97, gamma=4)
        grid_search_beta(device, args.backbone, args.method, args.dataset, alpha=0.8, gamma=4)
        grid_search_gamma(device, args.backbone, args.method, args.dataset, alpha=0.8, beta=0.97)
    elif args.dataset == 'douban-movie':
        grid_search_alpha(device, args.backbone, args.method, args.dataset, beta=0.85, gamma=4)
        grid_search_beta(device, args.backbone, args.method, args.dataset, alpha=0.95, gamma=4)
        grid_search_gamma(device, args.backbone, args.method, args.dataset, alpha=0.95, beta=0.85)
    elif args.dataset == 'ml-1m':
        grid_search_alpha(device, args.backbone, args.method, args.dataset, beta=0.99, gamma=4)
        grid_search_beta(device, args.backbone, args.method, args.dataset, alpha=0.95, gamma=4)
        grid_search_gamma(device, args.backbone, args.method, args.dataset, alpha=0.95, beta=0.99)

