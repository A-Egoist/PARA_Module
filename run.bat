@REM switch to anaconda environment
@REM conda activate GNN

set BASE_CMD=D:/anaconda3/envs/TWDP/python.exe d:/Projects/PARA_Module/main.py

%BASE_CMD% --backbone MF --method base --dataset ciao --mode both
%BASE_CMD% --backbone MF --method PARA --dataset ciao --mode both
@REM %BASE_CMD% --backbone LightGCN --method base --dataset ciao --mode both
@REM %BASE_CMD% --backbone LightGCN --method PARA --dataset ciao --mode both


@REM %BASE_CMD% --backbone MF --method base --dataset douban-book --mode both
@REM %BASE_CMD% --backbone MF --method PARA --dataset douban-book --mode both
@REM %BASE_CMD% --backbone LightGCN --method base --dataset douban-book --mode both
@REM %BASE_CMD% --backbone LightGCN --method PARA --dataset douban-book --mode both


@REM %BASE_CMD% --backbone MF --method base --dataset douban-movie --mode both
@REM %BASE_CMD% --backbone MF --method PARA --dataset douban-movie --mode test
@REM %BASE_CMD% --backbone LightGCN --method base --dataset douban-movie --mode both
@REM %BASE_CMD% --backbone LightGCN --method PARA --dataset douban-movie --mode test


@REM %BASE_CMD% --backbone MF --method base --dataset ml-1m --mode both
@REM %BASE_CMD% --backbone MF --method PARA --dataset ml-1m --mode test
@REM %BASE_CMD% --backbone LightGCN --method base --dataset ml-1m --mode both
@REM %BASE_CMD% --backbone LightGCN --method PARA --dataset ml-1m --mode test


echo All tasks are completed.

@REM shutdown -s -t 60