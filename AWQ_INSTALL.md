1. Clone awq repository in a separate folder and navigate to that folder
```
git clone https://github.com/mit-han-lab/llm-awq
cd llm-awq
```
2. Install Package
```
conda activate [name of wanda env]
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# install on cuda 11.7.1
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio -f https://download.pytorch.org/whl/cu117/torch_stable.html
export CUDA_HOME=/opt/apps/cuda/11.7.1
cd awq/kernels
python setup.py install
# to fix dependency issues:
pip install --upgrade datasets
pip install 'transformers<4.36'
```