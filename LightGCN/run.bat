echo 1/4
python main.py 2018-12-31_18 --embed_size 1024 --n_layers 1 --batch_size 20000 --epochs 40 --learning_rate 1e-05 --decay 0
echo 2/4
python main.py 2018-12-31_18 --embed_size 1024 --n_layers 1 --batch_size 20000 --epochs 100 --learning_rate 1e-05 --decay 0
echo 3/4
python main.py 2018-12-31_18 --embed_size 1024 --n_layers 1 --batch_size 20000 --epochs 70 --learning_rate 1e-05 --decay 0
echo 4/4
python main.py 2018-12-31_18 --embed_size 1024 --n_layers 1 --batch_size 20000 --epochs 50 --learning_rate 1e-05 --decay 0
