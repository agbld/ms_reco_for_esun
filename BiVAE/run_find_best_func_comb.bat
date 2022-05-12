@REM dataset_name_list = ['2018-12-31_18']
@REM latent_dim_list = [128]
@REM encoder_dims_list = [128]
@REM act_func_list = ['sigmoid', 'tanh']
@REM likelihood_list = ['bern', 'pois']
@REM num_epochs_list = [20, 40, 80, 120]
@REM batch_size_list = [1024]
@REM learning_rate_list = [0.002]

echo 1/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood bern --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 2/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood bern --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 3/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood bern --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 4/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood bern --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 5/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 6/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 7/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 8/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func sigmoid --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 9/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood bern --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 10/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood bern --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 11/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood bern --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 12/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood bern --num_epochs 120 --batch_size 1024 --learning_rate 0.002
echo 13/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood pois --num_epochs 20 --batch_size 1024 --learning_rate 0.002
echo 14/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood pois --num_epochs 40 --batch_size 1024 --learning_rate 0.002
echo 15/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood pois --num_epochs 80 --batch_size 1024 --learning_rate 0.002
echo 16/16
python main.py 2018-12-31_18 --latent_dim 128 --encoder_dims 128 --act_func tanh --likelihood pois --num_epochs 120 --batch_size 1024 --learning_rate 0.002
