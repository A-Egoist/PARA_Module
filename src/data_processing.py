import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
import pickle
import argparse
import swifter
import json


def load_data(dataset, train_data_path, valid_data_path, test_data_path):
    print(f'Loading dataset {dataset}.')
    train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    valid_data = pd.read_csv(valid_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    
    save_path = '.' + train_data_path.split('.')[1] + '.pkl'
    if os.path.exists(save_path):
        # load processed data
        with open(save_path, 'rb') as f:
            num_users, num_items, train_data, valid_data, test_data, item_information = pickle.load(f)
        return num_users, num_items, train_data, valid_data, test_data, item_information
    else:
        num_users = max(train_data['user'].max(), valid_data['user'].max(), test_data['user'].max()) + 1  # user id from 0 to max
        num_items = max(train_data['item'].max(), valid_data['item'].max(), test_data['item'].max()) + 1  # item id from 0 to max

        item_information = pd.DataFrame(range(num_items), columns=['item'])  # columns = ['item', 'count', 'popularity', 'quality']
        
        # count
        def item_count(row):
            return len(train_data[train_data['item'] == row['item']])
        item_information['count'] = item_information.swifter.apply(item_count, axis=1)  # C_{i}
        # item_information['count'].fillna(0, inplace=True)  # Set the number of items that have not appeared to 0
        item_count_max = item_information['count'].max()  # C_{Max}
        # item_count_min = item_information['count'].min()  # C_{Min} TODO
        print('Count finished.')

        # calc item popularity
        # TODO
        item_information['popularity'] = item_information['count'].swifter.apply(lambda x: np.round(x / item_count_max, 4))  # $m_{i}=\frac{C_{i}}{C_{Max}}$
        print('Calculate item popularity finished.')

        # calc item quality
        # Definition 6
        item_quality_6 = calculate_item_quality_method_6(train_data.copy())
        item_information = item_information.merge(item_quality_6, on='item', how='left')
        print('Calculate item quality finished.')
        
        item_information['popularity'].fillna(0, inplace=True)  # fill the NA with alpha
        item_information['quality'].fillna(0, inplace=True)  # fill the NA with beta
        print('Data processing finished.')

        # save processed data
        with open(save_path, 'wb') as f:
            pickle.dump((num_users, num_items, train_data, valid_data, test_data, item_information), f)
        return num_users, num_items, train_data, valid_data, test_data, item_information


def calculate_item_quality_method_6(train_data):
    # drop row with no rating
    train_data.drop(train_data[train_data['rating'] == -1.0].index, inplace=True)
    
    user_stddev = train_data.groupby('user')['rating'].std().reset_index()
    user_stddev.columns = ['user', 'rating_stddev']

    mu = user_stddev['rating_stddev'].median()
    sigma = user_stddev['rating_stddev'].std()

    def calculate_h_u(s_u, mu=mu, sigma=sigma):
        return np.exp(-((s_u - mu) ** 2) / (2 * sigma ** 2))

    user_stddev['h_u'] = user_stddev['rating_stddev'].swifter.apply(calculate_h_u)
    train_data = train_data.merge(user_stddev, on='user', how='left')
    train_data['h_u'].fillna(0, inplace=True)

    train_data['weighted_rating'] = train_data['h_u'] * train_data['rating']
    item_quality = train_data.groupby('item').apply(lambda x: 0 if np.sum(x['h_u']) == 0 else np.sum(x['weighted_rating']) / np.sum(x['h_u'])).reset_index()
    item_quality.columns = ['item', 'quality']
    
    # Max-Min Normalization
    q_min = item_quality['quality'].min()
    q_max = item_quality['quality'].max()
    item_quality['quality_normalized'] = (item_quality['quality'] - q_min) / (q_max - q_min)

    item_quality.drop(columns=['quality'], inplace=True)
    item_quality.rename(columns={'quality_normalized': 'quality'}, inplace=True)

    return item_quality


class BPRDataset(Dataset):
    def __init__(self, data, num_items, num_negatives=4, is_training=True):
        super().__init__()
        self.data = data  # DataFrame
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.is_training = is_training

    def negative_sample(self, dataset, extend_data_path):
        assert self.is_training, 'no need to sampling when testing'

        if dataset == 'ml-1m':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32, engine='python')
        elif dataset == 'douban-movie':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'amazon-music':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'ciao':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'douban-book':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'running-example':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        self.data_fill = extend_data.values.tolist()

    def __getitem__(self, index):
        data = self.data_fill if self.is_training else self.data.values.to_list()
        user = data[index][0]
        item_i = data[index][1]
        item_j = data[index][2] if self.is_training else data[index][1]
        return user, item_i, item_j
    
    def __len__(self):
        # TODO
        # return 29
        return self.num_negatives * len(self.data) if self.is_training else len(self.data)


def get_graph(dataset, num_users, num_items, train_data):
    if not os.path.exists('./data/LightGCN_graph'):
        os.makedirs('./data/LightGCN_graph')
    graph_index_path = './data/LightGCN_graph' + f'/{dataset}-graph_index.npy'
    graph_data_path = './data/LightGCN_graph' + f'/{dataset}-graph_data.npy'
    if os.path.exists(graph_data_path) and os.path.exists(graph_index_path):
        graph_index = np.load(graph_index_path)
        graph_data = np.load(graph_data_path)
        graph_index = torch.from_numpy(graph_index)
        graph_data = torch.from_numpy(graph_data)

        graph = torch.sparse.FloatTensor(graph_index, graph_data, torch.Size([num_users + num_items, num_users + num_items]))
        graph = graph.coalesce()
    else:
        trainUser = train_data['user'].values
        trainItem = train_data['item'].values
        user_dim = torch.LongTensor(trainUser)
        item_dim = torch.LongTensor(trainItem)

        # first subgraph
        first_sub = torch.stack([user_dim, item_dim + num_users])
        # second subgraph
        second_sub = torch.stack([item_dim + num_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()

        graph = torch.sparse.FloatTensor(index, data, torch.Size([num_users + num_items, num_users + num_items]))
        row_sum = torch.sparse.sum(graph, dim=1).to_dense()
        row_sum[row_sum == 0] = 1.
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        
        row, col = index
        data = data * d_inv_sqrt[row] * d_inv_sqrt[col]
        graph = torch.sparse.FloatTensor(index, data, torch.Size([num_users + num_items, num_users + num_items]))
        
        np.save(graph_index_path, index.numpy())
        np.save(graph_data_path, data.numpy())
        graph = graph.coalesce()
    return graph


def ciao_split():
    data_path = './data/Ciao/movie-ratings.txt'
    train_data_path = './data/Ciao/movie-ratings.train'
    valid_data_path = './data/Ciao/movie-ratings.valid'
    test_data_path = './data/Ciao/movie-ratings.test'
    data = pd.read_csv(data_path, sep=',', header=None, names=['user', 'item', 'genre', 'review', 'rating', 'timestamp'], usecols=[0, 1, 4, 5], dtype={'user': np.int32, 'item': np.int32, 'rating': np.int32, 'timestamp': np.str_})
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')
    
    # Before filtering statistics
    print("\n=== Before filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")

    # 5-core filtering
    user_counts = data['user'].value_counts()
    item_counts = data['item'].value_counts()
    data = data[(data['user'].isin(user_counts[user_counts >= 5].index)) & (data['item'].isin(item_counts[item_counts >= 5].index))]

    # Create new continuous IDs for users and items
    unique_users = data['user'].unique()
    unique_items = data['item'].unique()
    
    # Create mapping dictionaries
    user_id_map = {old: new for new, old in enumerate(unique_users)}
    item_id_map = {old: new for new, old in enumerate(unique_items)}
    
    # Map old IDs to new IDs
    data['user'] = data['user'].map(user_id_map)
    data['item'] = data['item'].map(item_id_map)

    # After filtering statistics
    print("\n=== After filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")

    total_len = len(data)
    train_end = int(total_len * 0.6)
    valid_end = int(total_len * 0.7)  # 6:1:3

    train_data = data.iloc[:train_end]
    valid_data = data.iloc[train_end:valid_end]
    test_data = data.iloc[valid_end:]

    train_data.to_csv(train_data_path, sep='\t', header=False, index=False)
    valid_data.to_csv(valid_data_path, sep='\t', header=False, index=False)
    test_data.to_csv(test_data_path, sep='\t', header=False, index=False)

    print(f"Ciao split done! Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")


def douban_book_split():
    data_path = './data/Douban/book/douban_book.tsv'
    train_data_path = './data/Douban/book/douban_book.train'
    valid_data_path = './data/Douban/book/douban_book.valid'
    test_data_path = './data/Douban/book/douban_book.test'
    data = pd.read_csv(data_path, sep='\t', header=None, skiprows=1, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3], dtype=np.int32)
    data.drop(data[data['rating']==-1].index, inplace=True)
    data = data.sort_values(by='timestamp')

    # Before filtering statistics
    print("\n=== Before filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")

    # 5-core filtering
    user_counts = data['user'].value_counts()
    item_counts = data['item'].value_counts()
    data = data[(data['user'].isin(user_counts[user_counts >= 5].index)) & (data['item'].isin(item_counts[item_counts >= 5].index))]

    # Create new continuous IDs for users and items
    unique_users = data['user'].unique()
    unique_items = data['item'].unique()
    
    # Create mapping dictionaries
    user_id_map = {old: new for new, old in enumerate(unique_users)}
    item_id_map = {old: new for new, old in enumerate(unique_items)}
    
    # Map old IDs to new IDs
    data['user'] = data['user'].map(user_id_map)
    data['item'] = data['item'].map(item_id_map)

    # After filtering statistics
    print("\n=== After filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")

    total_len = len(data)
    train_end = int(total_len * 0.6)
    valid_end = int(total_len * 0.7)  # 6:1:3

    train_data = data.iloc[:train_end]
    valid_data = data.iloc[train_end:valid_end]
    test_data = data.iloc[valid_end:]

    train_data.to_csv(train_data_path, sep='\t', header=False, index=False)
    valid_data.to_csv(valid_data_path, sep='\t', header=False, index=False)
    test_data.to_csv(test_data_path, sep='\t', header=False, index=False)

    print(f"Douban-book split done! Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")
    

def douban_movie_split():
    data_path = './data/Douban/movie/douban_movie.tsv'
    train_data_path = './data/Douban/movie/douban_movie.train'
    valid_data_path = './data/Douban/movie/douban_movie.valid'
    test_data_path = './data/Douban/movie/douban_movie.test'
    data = pd.read_csv(data_path, sep='\t', header=None, skiprows=1, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3], dtype=np.int32)
    data.drop(data[data['rating']==-1].index, inplace=True)
    data = data.sort_values(by='timestamp')

    # Before filtering statistics
    print("\n=== Before filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")
    
    # 10-core filtering
    user_counts = data['user'].value_counts()
    item_counts = data['item'].value_counts()
    data = data[(data['user'].isin(user_counts[user_counts >= 10].index)) & (data['item'].isin(item_counts[item_counts >= 10].index))]

    # Create new continuous IDs for users and items
    unique_users = data['user'].unique()
    unique_items = data['item'].unique()
    
    # Create mapping dictionaries
    user_id_map = {old: new for new, old in enumerate(unique_users)}
    item_id_map = {old: new for new, old in enumerate(unique_items)}
    
    # Map old IDs to new IDs
    data['user'] = data['user'].map(user_id_map)
    data['item'] = data['item'].map(item_id_map)

    # After filtering statistics
    print("\n=== After filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")

    total_len = len(data)
    train_end = int(total_len * 0.6)
    valid_end = int(total_len * 0.7)  # 6:1:3

    train_data = data.iloc[:train_end]
    valid_data = data.iloc[train_end:valid_end]
    test_data = data.iloc[valid_end:]

    train_data.to_csv(train_data_path, sep='\t', header=False, index=False)
    valid_data.to_csv(valid_data_path, sep='\t', header=False, index=False)
    test_data.to_csv(test_data_path, sep='\t', header=False, index=False)

    print(f"Douban-movie split done! Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")


def ml_1m_split():
    data_path = './data/ml-1m/ratings.dat'
    train_data_path = './data/ml-1m/ratings.train'
    valid_data_path = './data/ml-1m/ratings.valid'
    test_data_path = './data/ml-1m/ratings.test'
    data = pd.read_csv(data_path, sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3], dtype=np.int32, engine='python')
    data = data.sort_values(by='timestamp')

    # Before filtering statistics
    print("\n=== Before filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")

    # 5-core filtering
    user_counts = data['user'].value_counts()
    item_counts = data['item'].value_counts()
    data = data[(data['user'].isin(user_counts[user_counts >= 5].index)) & (data['item'].isin(item_counts[item_counts >= 5].index))]

    # Create new continuous IDs for users and items
    unique_users = data['user'].unique()
    unique_items = data['item'].unique()
    
    # Create mapping dictionaries
    user_id_map = {old: new for new, old in enumerate(unique_users)}
    item_id_map = {old: new for new, old in enumerate(unique_items)}
    
    # Map old IDs to new IDs
    data['user'] = data['user'].map(user_id_map)
    data['item'] = data['item'].map(item_id_map)

    # After filtering statistics
    print("\n=== After filtering ===")
    print(f"Total interactions: {len(data)}")
    print(f"Number of unique users: {data['user'].nunique()}")
    print(f"Number of unique items: {data['item'].nunique()}")
    print(f"Average interactions per user: {len(data)/data['user'].nunique():.2f}")
    print(f"Average interactions per item: {len(data)/data['item'].nunique():.2f}")

    total_len = len(data)
    train_end = int(total_len * 0.6)
    valid_end = int(total_len * 0.7)  # 6:1:3

    train_data = data.iloc[:train_end]
    valid_data = data.iloc[train_end:valid_end]
    test_data = data.iloc[valid_end:]

    train_data.to_csv(train_data_path, sep='\t', header=False, index=False)
    valid_data.to_csv(valid_data_path, sep='\t', header=False, index=False)
    test_data.to_csv(test_data_path, sep='\t', header=False, index=False)

    print(f"MovieLens-1M split done! Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(test_data)}")


def calc_sparse(dataset):
    if dataset == 'ciao':
        num_users = 2582
        num_items = 3066
        num_interactions = 33911
        density = num_interactions / num_users / num_items
        print(f'Sparsity: {np.round(1 - density, 4)}')  # Sparsity: 0.9957
    elif dataset == 'douban-book':
        num_users = 26431
        num_items = 30976
        num_interactions = 1116025
        density = num_interactions / num_users / num_items
        print(f'Sparsity: {np.round(1 - density, 4)}')  # Sparsity: 0.9986
    elif dataset == 'douban-movie':
        num_users = 52768
        num_items = 27309
        num_interactions = 8859540
        density = num_interactions / num_users / num_items
        print(f'Sparsity: {np.round(1 - density, 4)}')  # Sparsity: 0.9939
    elif dataset == 'ml-1m':
        num_users = 6040
        num_items = 3416
        num_interactions = 999611
        density = num_interactions / num_users / num_items
        print(f'Sparsity: {np.round(1 - density, 4)}')  # Sparsity: 0.9516


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--dataset', type=str, default='ciao', choices=['ciao', 'douban-book', 'douban-movie', 'ml-1m'])
    args = parser.parse_args()

    if args.dataset == 'ciao':
        ciao_split()
        calc_sparse('ciao')
    elif args.dataset == 'douban-book':
        douban_book_split()
        calc_sparse('douban-book')
    elif args.dataset == 'douban-movie':
        douban_movie_split()
        calc_sparse('douban-movie')
    elif args.dataset == 'ml-1m':
        ml_1m_split()
        calc_sparse('ml-1m')
