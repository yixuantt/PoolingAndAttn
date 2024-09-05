import torch
import transformers
import logging
from mteb import MTEB
from mteb_task import TASK_LIST_RETRIEVAL
import torch.nn.functional as F
import eval_instruction
from tqdm import tqdm
import numpy as np
from typing import Mapping, Dict, List
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel
from utils import create_batch_dict,move_to_cuda

# Change The Model Here
from model_loaders.model1 import EmbedModel
transformers.logging.set_verbosity_error()

base_model = "LLM Base Model"
adapter_path = "Lora Adapter"
output_dir = "Log File Path"
cache_dir = "Hugging Face Cache"

def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    return logger


logger = _setup_logger()

class RetrievalModel(DRESModel):
    def __init__(self, base_model, adapter_path, cache_dir, **kwargs):
        self.model = EmbedModel(base_model, adapter_path, cache_dir)
        self.gpu_count = torch.cuda.device_count()
        self.prompt = "Represent this sentence for searching relevant passages."
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path,
                                                       cache_dir = cache_dir,
                                                       add_eos_token = True)
        self.normalize = True        
        
        if self.gpu_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            
        self.model.cuda()
        self.model.eval()
            
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = [self.prompt + q for q in queries]

        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 64 * self.gpu_count
        for start_idx in tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]
            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)
            
            with torch.cuda.amp.autocast():
                embeddings = self.model(**batch_dict)
                encoded_embeds.append(embeddings.detach().cpu().numpy())
         
        return np.concatenate(encoded_embeds, axis=0)
    
    def set_prompt(self, prompt: str):
        self.prompt = prompt

def count_embed(task,model):
    print("TASK NAME: "+ task)
    p = eval_instruction.eval_name2instruct[task]
    model.set_prompt(prompt=p)
    logger.info('Set prompt: {}'.format(p))
    model.l2_normalize = True
    logger.info('Set l2_normalize to {}'.format(model.l2_normalize))
        
    evaluation = MTEB(
        tasks=[task],
        task_langs=["eng","en","eng-Latn"]
    )

    logger.info('Running evaluation for task: {}'.format(evaluation))
    eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
    if task == "FinancialPhrasebankClassification" :
        eval_splits = ["train"]
    evaluation.run(
        model, eval_splits=eval_splits,
        output_folder=f"{output_dir}/"
    )

if __name__ == '__main__':
    model_name = adapter_path.split("/")[-1]
    logger.info("*"*10+f"Running For Model {model_name}"+"*"*10)
    model = RetrievalModel(base_model,adapter_path,cache_dir)

    for task in TASK_LIST_RETRIEVAL:
        logger.info(f"Running task: {task}")
        count_embed(task,model)
