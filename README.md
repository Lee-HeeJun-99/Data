##### collect_datset_modes.m 실행해서 mode별 data 추출
##### export_dataset_for_python.m 추출한 mat 데이터 python 형식으로 변형
##### make_dataset.py 데이터 분리
##### make_dataset 옵션 예시 명령어 
##### python make_dataset.py --in_mat dataset_out/dataset_flat.mat --out_npz train_dataset.npz --win_len 256 --stride 64 --normalize zscore
##### make_dataset_stratified.py 모든 mode가 들어가게 나눠서 구성하는 데이터 분리 코드
##### 옵션 예시, seed는 0 ~ 순차적으로 생성, python make_dataset_stratified.py --in_mat .\dataset_flat.mat --out_npz .\train_seed0.npz --seed 0 --win_len 256 --stride 64 --normalize zscore --n_val_per_class 1 --n_test_per_class 1 
