import os
import json
import logging
import random
from pathlib import Path
from typing import List, Optional
from typing import Any, Dict, Tuple, Union

import datasets
import torch
import transformers
from dataclasses import dataclass, field
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments,AutoModel,Trainer
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from loss.HardNegativeNLLLoss import HardNegativeNLLLoss
logger = get_logger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The base model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The PEFT model checkpoint to add on top of base model.")},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use. Options: E5"},
    )
    dataset_file_path: Optional[str] = field(
        default=None, metadata={"help": "The input training data file or folder."}
    )
    # TODO: implement this
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )


@dataclass
class CustomArguments:
    """
    Custom arguments for the script
    """

    simcse_dropout: float = field(
        default=0.1, metadata={"help": "The SimCSE dropout rate for the model"}
    )

    lora_dropout: float = field(
        default=0.05, metadata={"help": "The dropout rate for lora"}
    )

    lora_r: int = field(default=8, metadata={"help": "The r value for lora"})

    cache_dir: Optional[str] = field(
        default= None, metadata={"help": "huggingface cache dir"}
    )

    loss_class: Optional[str] = field(
        default="HardNegativeNLLLoss",
        metadata={
            "help": "The loss class to use for training. Options: HardNegativeNLLLoss"
        },
    )

    loss_scale: float = field(
        default=50.0, metadata={"help": "The loss scale for the loss function"}
    )

    

def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
    )
    (model_args,
    data_args,
    training_args,
    custom_args) = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(kwargs_handlers=[])
    cache_dir = custom_args.cache_dir
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    # if args.seed is not None:
    #     set_seed(training_args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              cache_dir = cache_dir,
                                              add_eos_token =True)
    if not tokenizer.pad_token:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(data_args.dataset_name,
                           cache_dir = cache_dir)
    
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(dataset["train"])), 3):
        logger.info(f"Sample {index} of the training set: {dataset['train'][index]}.")
    
    class MistralForSequenceEmbedding(torch.nn.Module):
        def __init__(self,model_name_or_path):
            super(MistralForSequenceEmbedding, self).__init__()
            self.model = AutoModel.from_pretrained(model_name_or_path, 
                                                   cache_dir = cache_dir,
                                                   torch_dtype=torch.bfloat16,
                                                   attn_implementation="flash_attention_2",
                                                   output_hidden_states=True)
            peft_config = LoraConfig(**json.load(open("lora.json")))
            self.model = get_peft_model(self.model, peft_config)       

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """
            return_dict = return_dict if return_dict is not None else True

            transformer_outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = transformer_outputs.last_hidden_state
            embeddings = self.last_token_pool(hidden_states,attention_mask)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

        def last_token_pool(self, last_hidden_states, attention_mask):
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
    # base model
    model = MistralForSequenceEmbedding(model_args.model_name_or_path)
    # model.print_trainable_parameters()
    
    # import ipdb;ipdb.set_trace()
    accelerator.print(model)

    def collate_fn(batch):
        sentences, positives, negatives = zip(*batch)
        sentences = [item['query'] for item in batch]
        negatives = [item['negative'] for item in batch]
        positives = [item['positive'] for item in batch]

        result = []
        # Here you would typically convert sentences, positives, and negatives
        # to tensors. Since you haven't specified how you want to encode the text,
        # I'll leave that part out. You could use tokenization and numericalization,
        # for example, with a library like HuggingFace's `transformers`.

        sentence_batch_dict = tokenizer(sentences, max_length=model_args.max_seq_length, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        positives_batch_dict = tokenizer(positives, max_length=model_args.max_seq_length, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
        negatives_batch_dict = tokenizer(negatives, max_length=model_args.max_seq_length, return_attention_mask=True, padding=True, truncation=True, return_tensors='pt')
                                         
        result.append(sentence_batch_dict)
        result.append(positives_batch_dict)
        result.append(negatives_batch_dict)
        
        labels = [0] * len(sentences)
        return result,labels
    
    class MySupervisedTrainer(Trainer):

        def __init__(
            self,
            *args,
            loss_function=None,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.loss_function = loss_function
            self.train_dataloader = DataLoader(
                dataset['train'],
                shuffle=True,
                collate_fn=collate_fn,
                batch_size=self._train_batch_size,
                pin_memory=True,
            )

        def get_train_dataloader(self):
            return self.accelerator.prepare(self.train_dataloader)

        def compute_loss(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
            features, labels = inputs
            q_reps = self.model(**features[0])
            d_reps = self.model(**features[1])
            d_reps_neg = self.model(**features[2])
                
            loss = self.loss_function(q_reps, d_reps, d_reps_neg)
            return loss

        def _save(self, output_dir: Optional[str] = None, state_dict=None):
            # If we are executing this function, we are the process zero, so we don't check for that.
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")

    #         self.model.save(output_dir)

            self.model.model.save_pretrained(output_dir)                
            self.tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    loss_fn = HardNegativeNLLLoss()
    trainer = MySupervisedTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        loss_function=loss_fn,
    )

    trainer.train()
    model.model.save_pretrained(training_args.output_dir)      
    torch.save(model.attn_pool.state_dict(),training_args.output_dir+"attn_pool.pt")
    tokenizer.save_pretrained(training_args.output_dir)
    
if __name__ == "__main__":
    main()