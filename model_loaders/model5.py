
import torch
from tqdm import tqdm
import numpy as np
from typing import Mapping, Dict, List
from transformers import AutoModel,AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from typing import List, Optional
from models.bidirectional_mistral import MistralBiModel
from pooling_layers.MultiLayersTrainablePooling import PerceiverResampler

class EmbedModel(torch.nn.Module):
    def __init__(self, base_model, adapter_path, cache_dir, **kwargs):
        super().__init__()
        self.model = MistralBiModel.from_pretrained(base_model, 
                                               torch_dtype=torch.bfloat16,
                                               output_hidden_states=True,
                                               attn_implementation = "flash_attention_2",
                                               cache_dir = cache_dir)
        self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                torch_dtype=torch.bfloat16,
        )
        self.prompt = "Represent this sentence for searching relevant passages."
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path,
                                                       cache_dir = cache_dir,
                                                       add_eos_token = True)
        self.normalize = True 

        self.attn_pool = PerceiverResampler(dim = self.model.config.hidden_size,
                                            hidden_dim=self.model.config.hidden_size,
                                            latent_dim=self.model.config.hidden_size)
        self.attn_pool.load_state_dict(torch.load(f"{adapter_path}/attn_pool.pt",map_location=torch.device('cpu')))
        
        
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

            hidden_states = transformer_outputs.hidden_states 
            layers = len(hidden_states)-1
            attention_mask = (input_ids != self.tokenizer.pad_token_id)
            extended_attention_mask = attention_mask.unsqueeze(-1).bool()
            avg_token_hidden_states = torch.stack([
                torch.sum(hidden_states[layer] * extended_attention_mask, dim=1) / torch.sum(extended_attention_mask, dim=1)
                for layer in range(1, layers + 1)
            ], dim=1)
            embeddings = self.attn_pool(avg_token_hidden_states)

            return embeddings