from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import sys
import json
import os.path as osp
from typing import Union
import torch
import math
import statistics
from tqdm import tqdm

load_8bit: bool = False
base_model= 'meta-llama/Llama-2-7b-hf'
lora_weights: str = "path/of/lora"
test_file = "example.json"
output_file = "example.txt"
max_new_tokens = 1
perturbation_times = 1

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = f"{template_name}.json"
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


prompt_template: str = "alpaca",
y_true  = []
y_label1  = []
y_score1  = []
y_label2  = []
y_score2  = []
stdevp = []
acc_num = 0
i=0
perturbation_results = []
prompter = Prompter()
tokenizer = LlamaTokenizer.from_pretrained(base_model)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )

model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0,
    top_p=0.75,
    top_k=40,
    num_beams=2,
    max_new_tokens=max_new_tokens,
    perturbation=False,
    **kwargs):

    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(device)

    if(perturbation):
        import random
        token_to_delete_index = random.randint(1, len(input_ids[0])-7)  # avoid deleting special tokens
        
        input_ids = torch.cat((input_ids[:,:token_to_delete_index], input_ids[:,token_to_delete_index+1:]), axis = 1).to(device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_beams,
        num_beams=num_beams,
        **kwargs,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    ss = generation_output.sequences
    s = ss[0]

    global scores,sqss
    scores = generation_output.scores
    sqss = tuple(
        prompter.get_response(
            tokenizer.decode(ss[i], skip_special_tokens=True)
        ) for i in range(num_beams))

    output = tokenizer.decode(s,skip_special_tokens=True)
    
    yield prompter.get_response(output)
with open(test_file, "r", encoding="utf-8") as f_in:
    data = json.load(f_in)
    for line in tqdm(data):
        perturbation_result = []
        for n in range(perturbation_times+1):
            perturbation = False if(n==0) else True
            input1 = line["input"]
            label = line["output"]
            instruction = line["instruction"]
            output = next(evaluate(instruction=instruction,input=f"{input1}",perturbation=perturbation))
            if(str(output).replace("</s>","")==str(label)):
                acc_num+=1
            i+=1
            if(max_new_tokens==1):
                if(n==0):
                    y_true.append(label)
                    token_scores, tokens = torch.topk(scores[0][0], 2, dim=-1)
                    y_label1.append(tokenizer.decode(tokens[0], skip_special_tokens=True))
                    y_score1.append(math.exp(token_scores[0]) / (math.exp(token_scores[0]) + math.exp(token_scores[1])))
                    y_label2.append(tokenizer.decode(tokens[1], skip_special_tokens=True))
                    y_score2.append(math.exp(token_scores[1]) / (math.exp(token_scores[0]) + math.exp(token_scores[1])))
                else:
                    token_scores , tokens = torch.topk(scores[0][0], 2, dim=-1)
                    perturbation_result.append(math.exp(token_scores[0]) / (math.exp(token_scores[0]) + math.exp(token_scores[1])))
            elif(max_new_tokens==2):
                top_fathers, _ = torch.topk(scores[0][0], 2, dim=-1)
                probabilities = []
                for k in range(2):
                    father_score = top_fathers[k]
                    self_score, self_tokens = torch.topk(scores[1][k], 2, dim=-1)
                    for j in range(2):
                        probabilities.append(father_score + self_score[j])
                probabilities.sort()
                probabilities.reverse()
                if (n == 0):
                    y_true.append(label)
                    y_label1.append(sqss[0])
                    y_score1.append(
                        math.exp(probabilities[0]) / (math.exp(probabilities[0]) + math.exp(probabilities[1])))
                    y_label2.append(sqss[1])
                    y_score2.append(
                        math.exp(probabilities[1]) / (math.exp(probabilities[0]) + math.exp(probabilities[1])))
                else:
                    perturbation_result.append(
                        math.exp(probabilities[0]) / (math.exp(probabilities[0]) + math.exp(probabilities[1])))

        stdevp.append(statistics.pstdev(perturbation_result))
        perturbation_results.append(perturbation_result)
with open(output_file, 'w', encoding='utf-8') as file:
    for item1,item2,item3,item4,item5,item6,item7 in zip(y_true, y_label1, y_score1, y_label2, y_score2, stdevp, perturbation_results):
        file.write(f'{item1}\t{item2}\t{item3}\t{item4}\t{item5}\t{item6}\t{item7}\n')