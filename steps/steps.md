conda env create --prefix ./env -f environment.yml

To clean up or delete the env:
conda remove --prefix ./env --all -y


# To activate this environment, use
#     $ conda activate E:\CICAPT_IIoT_GIN\env
#     $ conda activate ./env
#
# To deactivate an active environment, use
#
#     $ conda deactivate



pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html 

pip install scikit-learn==1.2.2 xgboost==1.7.5 imbalanced-learn==0.10.1 tensorboard==2.13.0


pip install matplotlib==3.7.1 seaborn==0.12.2
pip install numpy==1.24.4
pip install dgl-2.0.0+cu118-cp310-cp310-win_amd64.whl
