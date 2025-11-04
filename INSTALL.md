# Installation  
Step 1: Create a new conda environment:
```bash
conda create -n prune_llm python=3.13 -y
conda activate prune_llm
```

Step 2: Install cuda-toolkit
```bash
conda install cuda-toolkit
```

Step 2.1: Check cuda-toolkit installation
```bash
which nvcc
nvcc --version
```

Step 2.2: Set `CUDA_HOME`
```bash
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
```

Step 3: Install relevant packages
For DGX Spark:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install transformers datasets wandb sentencepiece accelerate
```

For Others:
```bash
pip install torch torchvision torchaudio transformers datasets wandb sentencepiece accelerate
```

Step 4: Check if CUDA is available.
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```