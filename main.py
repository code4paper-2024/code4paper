from typing import Any, List, Union
import json
from finetune import train
import os.path as osp
prompt_template: str = "alpaca"

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

def generate_full_prompt(data_point):
    instruction = data_point['instruction']
    input = data_point['input']
    output = data_point['output']
    prompter = Prompter()
    prompt = prompter.generate_prompt(instruction, input)
    completion = output

    return {
        'prompt': prompt,
        'completion': completion,
        '_var_instruction' : instruction,
        '_var_input' : input
    }

if __name__ == "__main__":
    base_model = "meta-llama/Llama-2-7b-hf"
    tokenizer = "meta-llama/Llama-2-7b-hf"
    output_dir = "path/to/output"
    train_data = "example.json"
    val_data = "example.json"
    
    final_train_data = []
    with open(train_data, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
        for line in data:
            final_train_data.append(generate_full_prompt(line))
            
    final_val_data = []
    with open(val_data, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
        for line in data:
            final_val_data.append(generate_full_prompt(line))

    train(
        base_model=base_model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        train_data=final_train_data,
        load_in_8bit=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=False,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
        learning_rate=3e-4,
        cutoff_len=512,
        val_data=final_val_data,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_modules=["q_proj", "v_proj"],
        lora_modules_to_save=[],
        train_on_inputs=True,
        group_by_length=False,
        save_steps=500,
        save_total_limit=3,
        logging_steps=10,
        hf_access_token=None,
    )