import os
os.environ["CUDA_VISIBLE_DEVICES"] = input("Please input CUDA_VISIBLE_DEVICES: ")
from typing import List

from vllm import LLM, SamplingParams
from vllm.inputs import PromptInputs
import typing
import pandas as pd
import tqdm.auto as tqdm

llm = LLM(
    model="/data/share_weight/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1,
    use_v2_block_manager=True,
    classifier_free_guidance_model="/data/share_weight/Meta-Llama-3-8B-Instruct",
)

tokenizer = llm.get_tokenizer()
def generate_prompt(sents: str | typing.List[str]):
    if isinstance(sents, str):
        sents = [sents]

    cfg_prompt = [
        str(
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"""\
                            Rewrite the following sentence or phrase using different words and sentence structure while preserving its original meaning. 
                            Directly answer with the rewritten sentence. Don't give any explanation or description other than the rewritten sentence.
                            Write a sentence that is entailment with: ```{sent.strip()}```. Result: """,
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        for sent in sents
    ]
    return cfg_prompt

def generate_sent(sent, batch_size, gen_num=32):
    results = []
    results_broken = []
    for _ in range(batch_size):
        results.append([])
        results_broken.append(0)
    
    last_sent = sent
    current_sent = sent
    for round in range(gen_num):
        cfg_prompt = generate_prompt(current_sent)
        base_prompt = generate_prompt(last_sent)
        sampling_params = SamplingParams(
            temperature=0, 
            guidance_scale=1.5,
            skip_special_tokens=True,
            max_tokens = 100,
        )
        inputs = [{"prompt": cfg_prompt[i], "negative_prompt": base_prompt[i]} for i in range(len(cfg_prompt))]
        
        outputs = llm.generate(inputs, sampling_params)
        outputs = [output.outputs[0].text for output in outputs]
        outputs = [output.strip('`').strip() for output in outputs]

        last_sent = current_sent
        current_sent = outputs
        # print(f"Round {round} => {json.dumps(current_sent, indent=4, ensure_ascii=False)}")

        for idx, output in enumerate(outputs):
            if '\n' in output:
                results_broken[idx] = 1
        for idx, ns in enumerate(outputs):
            results[idx].append(ns)

    return results, results_broken

sentences = []

# read the selected sentences
with open("../data/selected_sentences.txt", 'r', encoding='utf-8') as file:
    for line in file:
        sentence = line.strip()
        sentences.append(sentence)

batch_size = 4
# set the start and end index of the sentences to generate
# you can create multiple processes to generate the sentences by setting different start and end index
st = 0
end = 40000

each_data_num = 25
save_data = pd.DataFrame(columns=("sent", "round"))

for i in range(int((end-st)/each_data_num)):
    print("the {} th round begin...".format(i))
    for batch_idx in tqdm.trange(each_data_num):
        sents = sentences[st + i*each_data_num + batch_idx:st + i*each_data_num + (batch_idx+1)]
        results, results_broken = generate_sent(sents, batch_size)
        for sent, result, rb in zip(sents, results, results_broken):
            if rb == 1:
                continue
            row = pd.DataFrame({'sent': [sent], 'round': [0]})
            save_data = pd.concat([save_data, row], ignore_index=True)
            for round, r in enumerate(result):
                row = pd.DataFrame({'sent': [r], 'round': [round+1]})
                save_data = pd.concat([save_data, row], ignore_index=True)
                round += 1
    # save the generated ranking senteces
    save_data.to_csv("../data/ranking_sentences_1.csv")