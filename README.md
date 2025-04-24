# PARA_Module
This is the PyTorch implementation of our paper "Popularity--quality Driven Rating Prediction Adjustment with Trisecting-acting-outcome."

![The Workflow of PARA](img/PARA-workflow.svg)

## Requirements

*   numpy=1.21.5
*   pandas=1.3.5
*   python=3.7.16
*   pytorch=1.13.1

## Datasets

| Dataset      |  Users |   Items | Interactions |                             Link                             |
| ------------ | -----: | ------: | -----------: | :----------------------------------------------------------: |
| Ciao         | 17 615 |  16 121 |       72 665 |   [URL](https://guoguibing.github.io/librec/datasets.html)   |
| Douban-Book  | 46 548 | 212 995 |    1 908 081 | [URL](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data) |
| Douban-Movie | 94 890 |  81 906 |   11 742 260 | [URL](https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/README.md#douban-data) |
| MovieLens-1M |  6 040 |   3 900 |    1 000 209 |     [URL](https://grouplens.org/datasets/movielens/1m/)      |

## Simply Reproduce the Results

Take the Ciao dataset as an example:

1.   Clone the source code

     ```bash
     git clone https://github.com/A-Egoist/PARA_Module.git --depth=1
     ```

4.   Run inference
     To evaluate the model, run the following command:

     ```bash
     # Windows
     python .\main.py --backbone MF --method PARA --dataset ciao --mode test
     ```

     **Avaliable Options:**

     *   `--backbone`: The backbone model. Available options: `['MF', 'LightGCN']`.
     *   `--method`: The method to be used. Available options: `['Base', 'PARA']`.
     *   `--dataset`: The dataset to use. Available options: `['ciao', 'douban-book', 'douban-movie', 'ml-1m]`.
     *   `--mode`: The mode to be choosen. Available options: `['train', 'test', 'both']`.

## Start from Scratch

This section explains how to reproduce the results from scratch, taking the Ciao dataset as an example:

1.   **Clone source code and datasets**
     Clone the repository containing the code:

     ```bash
     git clone https://github.com/A-Egoist/TWDP.git --depth=1
     ```

     Download the Ciao dataset from [URL](https://guoguibing.github.io/librec/datasets.html) and move it into the `data` folder.

2.   **Data preprocessing**

     (a). Split the dataset chronologically into training (60%), validation (10%), and test (30%) sets.

     Run the following command to split the Ciao dataset:

     ```python
     # Windows
     python .\src\data_processing.py --dataset ciao
     ```

     **Available Option:**

     *   `--dataset`: The dataset to be splited. Available options: `['ciao', 'douban-book', 'douban-movie', 'ml-1m']`.

     (b). Compile the negative sampling script

     Use the following command to compile the C++ script for negative sampling:

     ```bash
     # Windows
     g++ .\src\negative_sampling.cpp -o .\src\negative_sampling.exe
     ```

     (c). Perform negative sampling

     Execute the compiled script to perform negative sampling:

     ```bash
     # Windows
     .\src\negative_sampling.exe ciao
     ```

     **Explanation of The Parameter:**

     *   The parameter specifies the dataset to be processed, with available options: `['ciao', 'douban-book', 'douban-movie', 'ml-1m']`.

3.   **Training and Inference**
     To start the training process, run the following command:

     ```bash
     # Windows
     python .\main.py --backbone MF --method PARA --dataset ciao --mode both
     ```

     **Available Options:**

     *   `--backbone`: The backbone model. Available options: `['MF', 'LightGCN']`.
     *   `--method`: The method to be used. Available options: `['Base', 'PARA']`.
     *   `--dataset`: The dataset to use. Available options: `['ciao', 'douban-book', 'douban-movie', 'ml-1m']`.
     *   `--mode`: The mode to be chosen. Available options: `['train', 'test', 'both']`.

4.   **Results**
     The results are saved in the `logs/ciao` folder.

## Citation

If you use this code, please cite the following paper:

```bibtex
@article{ZhangLong2025PARA,
  title   = {Plug-and-play Rating Prediction Adjustment through Trisecting-acting-outcome},
  author  = {},
  journal = {},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

## Acknowledgments
