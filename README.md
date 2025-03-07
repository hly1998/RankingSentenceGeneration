# Refining Sentence Embedding Model through Ranking Sentences Generation with Large Language Models

This is the code implementation for the paper *Refining Sentence Embedding Model through Ranking Sentences Generation with Large Language Models*, encompassing data synthesis, model execution, and model validation.

The paper is now avaliable at: https://arxiv.org/pdf/2502.13656

The environments differ across various stages. For details on the data synthesis phase, please refer to *./generation/README.md*.  

Below, we introduce the environment configuration for the post-training phase. In this stage, we draw upon the implementations of **SimCSE** and **RankCSE**, utilizing a largely identical environment setup.

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org). 

```bash
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

If you instead use **CUDA** `<11` or **CPU**, install PyTorch by the following command,

```bash
pip install torch==1.7.1
```

Then run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

### Download the pretraining dataset
```
cd data
bash download_wiki.sh
```

### Download the downstream dataset
```
cd SentEval/data/downstream/
bash download_dataset.sh
```

## Post-Training
(The same as `run.sh`.)
```bash
CUDA_VISIBLE_DEVICES=6 python train.py \
    --model_name_or_path checkpoints/syn-bert-base-uncased \
    --train_file /data/rankingsentences_n:32.csv \
    --output_dir checkpoints/syn-bert-base-uncased-r-lr:3e-6-es:25-dw:0.5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 3e-6 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 25 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --omega 0.5 \
    --do_train \
    --do_eval \
    --fp16 \
    --teacher_name_or_path /data/lyhe/data/diffcse-bert-base-uncased-sts
```

Post-training arguments:
* `teacher_name_or_path`: The model checkpoint for weights of the teacher model.
* `--omega`: The weight for control the ranking information. same as omega in paper.
* `--train_file`: The ranking sentences dataset path.
* `--model_name_or_path`: Pre-trained checkpoints for the embedding model, i.e., `pcl-bert-base-uncased`

Arguments from [SimCSE](https://github.com/princeton-nlp/SimCSE):
* `--pooler_type`: Pooling method.
* `--mlp_only_train`: For unsupervised SimCSE-based models, it works better to train the model with MLP layer but test the model without it.

## Evaluation

In our experiments, the three tasks—**STS, Reranking, and TR**—involve two benchmarks: **SentEval** and **METB**.  

You can execute the following commands to evaluate a model on the **STS** and **TR** tasks.

```bash
python evaluation.py \
    --model_name_or_path <your_output_model_dir> \
    --pooler cls_before_pooler \
    --task_set <sts|transfer|full> \
    --mode test
```

You can use the *evaluation_mteb.py* file to validate the four **Reranking** datasets.  

Before running it, you may need to install the **MTEB** library by executing the following command:
```bash
pip install mteb
```

Modify the *evaluation_mteb.py* file to update the model names in **model_name_list**, then execute *evaluation_mteb.py*.

## Dataset

We provide our generated ranked sentence data for further research. At the same time, we offer synthetic data based on the MultiCSR-provided methodology.

[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow)](https://huggingface.co/leoner24)

* [leoner24/RankingSentences-NLI-LLaMA3-8B-32](https://huggingface.co/datasets/leoner24/RankingSentences-NLI-LLaMA3-8B-32)
* [leoner24/MultiCSR_NLI](https://huggingface.co/datasets/leoner24/MultiCSR_NLI) 

## Pretrained models

We provide our model trained based on MultiCSR and SynCSE, which has achieved state-of-the-art results. Additionally, we offer the checkpoints obtained using MultiCSR and SynCSE.

[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/leoner24)

Our post-training models:

* [leoner24/MultiCSR-r-BERT-base](https://huggingface.co/leoner24/MultiCSR-r-BERT-base)
* [leoner24/MultiCSR-r-BERT-large](https://huggingface.co/leoner24/MultiCSR-r-BERT-large)
* [leoner24/MultiCSR-r-RoBERTa-base](https://huggingface.co/leoner24/MultiCSR-r-RoBERTa-base)
* [leoner24/SynCSE-r-RoBERTa-large](https://huggingface.co/leoner24/SynCSE-r-RoBERTa-large)

MultiCSR (unofficial):
* [leoner24/multicse-bert-base-uncased](https://huggingface.co/leoner24/multicse-bert-base-uncased)
* [leoner24/multicse-bert-large-uncased](https://huggingface.co/leoner24/multicse-bert-large-uncased)
* [leoner24/multicse-roberta-base](https://huggingface.co/leoner24/multicse-roberta-base)
* [leoner24/multicse-roberta-large](https://huggingface.co/leoner24/multicse-roberta-large)

SynCSE (unofficial):
* [leoner24/syncse-bert-base-uncased](https://huggingface.co/leoner24/syncse-bert-base-uncased)
* [leoner24/syncse-bert-large-uncased](https://huggingface.co/leoner24/syncse-bert-large-uncased)

## Citations

Please cite our paper if you use it:

```bibtex
@article{he2025refining,
  title={Refining Sentence Embedding Model through Ranking Sentences Generation with Large Language Models},
  author={He, Liyang and Liu, Chenglong and Li, Rui and Huang, Zhenya and Ruan, Shulan and Zhou, Jun and Chen, Enhong},
  journal={arXiv preprint arXiv:2502.13656},
  year={2025}
}
```