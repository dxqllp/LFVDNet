python main.py --data_path Data/ --d_model 64 --d_inner_hid 64 --n_head 4 --n_layers 2 --d_k 16 --d_v 4 --lr 0.001 --task 'Phsyio4000'  --test_only True

python main.py --data_path Data/ --d_model 64 --d_inner_hid 64 --n_head 2 --n_layers 3 --d_k 16 --d_v 8--lr 0.0005 --task 'Phsyio8000'  --test_only True

python main.py --data_path Data/ --d_model 64 --d_inner_hid 64 --n_head 1 --n_layers 1 --d_k 8 --d_v 32--lr 0.0005 --task 'Phsyio12000'  --test_only True

python main.py --data_path Data/mimic-iv-2.2/ --d_model 32 --d_inner_hid 32 --n_head 1 --n_layers 2 --d_k 4 --d_v 8--lr 0.001 --task 'MIMIC-IV'  --test_only True
