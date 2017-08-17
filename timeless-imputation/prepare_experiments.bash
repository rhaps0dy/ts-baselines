python3 -m venv /home/scratch/ms16aga/venv3
. ~/.bashrc
cd ~/MIMIC/ts-baselines/timeless-imputation/
pip install --upgrade pip
pip install -r requirements.txt
pip install GPy
mkdir /home/scratch/ms16aga/impute_benchmark
#ln -s /home/scratch/ms16aga/impute_benchmark .
cp sklearn_base.py /home/scratch/ms16aga/venv3/lib64/python3.5/site-packages/sklearn/mixture/base.py
