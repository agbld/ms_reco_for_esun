@REM dataset_name_list = ['2018-12-31_18']
@REM latent_dim_list = [64, 128, 256]
@REM encoder_dims_list = [64, 128, 256]
@REM act_func_list = ['sigmoid']
@REM likelihood_list = ['pois']
@REM num_epochs_list = [20, 40, 80, 120]
@REM batch_size_list = [1024]
@REM learning_rate_list = [0.001, 0.002, 0.004]

echo 1/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 2/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 3/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 4/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 5/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 6/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 7/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 8/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 9/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 10/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 11/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 12/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 13/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 14/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 15/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 16/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 17/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 18/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 19/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 20/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 21/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 22/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 23/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 24/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 25/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 26/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 27/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 28/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 29/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 30/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 31/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 32/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 33/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 34/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 35/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 36/108
python main.py 2018-12-31_18 --latent_dim 64 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 37/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 38/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 39/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 40/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 41/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 42/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 43/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 44/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 45/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 46/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 47/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 48/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 49/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 50/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 51/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 52/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 53/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 54/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 55/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 56/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 57/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 58/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 59/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 60/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 61/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 62/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 63/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 64/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 65/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 66/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 67/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 68/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 69/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 70/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 71/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 72/108
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 73/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 74/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 75/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 76/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 77/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 78/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 79/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 80/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 81/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 82/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 83/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 84/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 64 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 85/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 86/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 87/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 88/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 89/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 90/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 91/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 92/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 93/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 94/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 95/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 96/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
echo 97/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.001
echo 98/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 99/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.004
echo 100/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.001
echo 101/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 102/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.004
echo 103/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.001
echo 104/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 105/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.004
echo 106/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.001
echo 107/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 108/108
python main.py 2018-12-31_18 --latent_dim 256 --encoder_dims 256 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.004
