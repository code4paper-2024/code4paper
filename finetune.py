import os
import sys
import re
from typing import Any, List, Union

import json
import torch
import transformers
from datasets import Dataset, load_dataset


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


def train(
    base_model: Any,
    tokenizer: Any,
    output_dir: str,
    train_data: List[Any], 
    val_data: List[Any],
    load_in_8bit=False,
    fp16=True,
    bf16=False,
    gradient_checkpointing=False,
    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 32,
    num_train_epochs: int = 5,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj","v_proj",],
    lora_modules_to_save: Union[List[str], None] = [],
    train_on_inputs: bool = True,
    group_by_length: bool = False,
    save_steps: int = 200,
    save_total_limit: int = 3,
    logging_steps: int = 10,
    additional_training_arguments: Union[dict, str, None] = None,
    additional_lora_config: Union[dict, str, None] = None,
    callbacks: List[Any] = [],
    hf_access_token: Union[str, None] = None,
):

    device_map = "auto"
    model = base_model
    if isinstance(model, str):
        model_name = model
        print(f"Loading base model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16,
            llm_int8_skip_modules=lora_modules_to_save,
            device_map=device_map,
            use_auth_token=hf_access_token
        )
        if re.match("[^/]+/llama", model_name):
            print(f"Setting special tokens for LLaMA model {model_name}...")
            model.config.pad_token_id = 0
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

        print(f"Loaded model {model_name}")

    if isinstance(tokenizer, str):
        tokenizer_name = tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_auth_token=hf_access_token
        )

        if re.match("[^/]+/llama", tokenizer_name):
            print(
                f"Setting special tokens for LLaMA tokenizer {tokenizer_name}...")
            tokenizer.pad_token_id = 0
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2

        print(f"Loaded tokenizer {tokenizer_name}")

    tokenizer.padding_side = "left"

    model = prepare_model_for_int8_training(model) # if load_in_8bit=false, nothing will happen

    lora_config_args = {
        'r': lora_r,
        'lora_alpha': lora_alpha,
        'target_modules': lora_target_modules,
        'modules_to_save': lora_modules_to_save,
        'lora_dropout': lora_dropout,
        'bias': "none",
        'task_type': "CAUSAL_LM",
    }
    config = LoraConfig(**{
        **lora_config_args,
        **(additional_lora_config or {}),
    })
    model = get_peft_model(model, config)
    if bf16:
        model = model.to(torch.bfloat16)

    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params} (calculated)")
    model.print_trainable_parameters()

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point["prompt"] + data_point["completion"]
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = data_point["prompt"]
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

    train_data = Dataset.from_list(train_data)
    val_data = Dataset.from_list(val_data)

    train_data = (
        train_data.shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        val_data.shuffle().map(generate_and_tokenize_prompt)
    )

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = {
        'output_dir': output_dir,
        'per_device_train_batch_size': micro_batch_size,
        'gradient_checkpointing': gradient_checkpointing,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'warmup_steps': 100,
        'num_train_epochs': num_train_epochs,
        'learning_rate': learning_rate,
        'fp16': fp16,
        'bf16': bf16,
        'logging_steps': logging_steps,
        'optim': "adamw_torch",
        'evaluation_strategy': "steps",
        'save_strategy': "steps",
        'eval_steps': save_steps,
        'save_steps': save_steps,
        'output_dir': output_dir,
        'save_total_limit': save_total_limit,
        'load_best_model_at_end': True,
        'group_by_length': group_by_length,
    }
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        args=transformers.TrainingArguments(**{
            **training_args,
            **(additional_training_arguments or {})
        }),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    train_output = trainer.train()

    model.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}.")
    return train_output
