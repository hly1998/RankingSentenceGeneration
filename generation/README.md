# Ranking Sentence Generation

This section explains how to generate the ranking list. We utilize vLLM to accelerate the generation process, significantly enhancing its speed. Moreover, the generation process differs from the post-training environment. To begin, follow the installation instructions below to configure the environment.

## Install
```bash
echo > ~/.condarc <<EOF
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/main
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/r
  - https://mirrors.ustc.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.ustc.edu.cn/anaconda/cloud
  pytorch: https://mirrors.ustc.edu.cn/anaconda/cloud
EOF

conda create -n sentgen python=3.11 -y
conda activate sentgen

pip install -i https://mirrors.ustc.edu.cn/pypi/simple pip -U
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple

pip install librosa ipykernel ipywidgets -v
pip install vllm-0.5.4-cp311-cp311-linux_x86_64.whl
```

We use the vLLM version **vllm-0.5.4-cp311-cp311-linux_x86_64.whl**. Newer versions may introduce API changes, which would require modifications to **generation.py**. Therefore, we recommend using the same version as the one employed in our execution to ensure compatibility.

First, you need to create a **sentences** file (**.txt**) containing the sentences to be processed, with each line representing a single sentence. For example, you can extract the premises from an **NLI dataset** as the input sentences. Alternatively, as demonstrated in our article, you may employ specific methods to extract partial sentences for generation. Then, execute the command to begin the process.

You can adjust the **batch size** according to your GPU environment to control the generation speed. Additionally, you may run multiple instances of the program simultaneously by modifying the **start (st)** and **end (end)** parameters to allocate different execution ranges.

Finally, execute **sentences_merge.py** to merge multiple (or single) generated ranking sentence results. The script will process them into a list format suitable for model training while also applying normalization procedures.

**Important:** We are using the **CUDA 12.4** environment, which is compatible with **vllm-0.5.4-cp311-cp311-linux_x86_64.whl**. If your CUDA version is not **12.4**, you may need to install a different **vLLM** version that matches your setup.