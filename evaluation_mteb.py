import os 
os.environ["CUDA_VISIBLE_DEVICES"] = input("Please input CUDA_VISIBLE_DEVICES: ")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from mteb import MTEB
from sentence_transformers import SentenceTransformer

# input the model name
model_name_list = [
                   "longgen-soft_sort-syn-roberta-large-uncased-lr:3e-6-es:25-dw:0.5",
                   ]

for model_name in model_name_list:
    model_pth = "./checkpoints/" + model_name
    model = SentenceTransformer(model_pth)
    evaluation = MTEB(tasks=["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR", "StackOverflowDupQuestions"], model_name=model_name, task_langs=["en"])
    results = evaluation.run(model, output_folder="./mteb_results/" + model_name)