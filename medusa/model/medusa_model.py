import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = 5 # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.medusa):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
    

    def adaptive_based_on_previous_step(self, past_accepted_count, max_heads=None):
        """
        Dynamically adjust number of heads based on acceptance rate in previous step.
        
        Args:
            past_accepted_count: Number of tokens accepted in previous step
            max_heads: Maximum number of heads to use (defaults to self.medusa)
        
        Returns:
            optimal_heads: Optimal number of heads to use for next decoding step
        """
        if max_heads is None:
            max_heads = self.medusa
            
        # Simple heuristic: Use number of tokens accepted in previous step
        # with some constraints to stay within reasonable bounds
        optimal_heads = min(max(past_accepted_count, 2), max_heads)
        return optimal_heads
    
    def adaptive_based_on_current_step(self, candidates_prob, threshold=0.5, max_heads=None):
        """
        Dynamically adjust number of heads during current step based on token probability.
        
        Args:
            candidates_prob: Probabilities of candidate tokens
            threshold: Probability threshold for acceptance
            max_heads: Maximum number of heads to use (defaults to self.medusa)
        
        Returns:
            prefix_length: Length of trusted prefix to use
        """
        if max_heads is None:
            max_heads = self.medusa
            
        # Find the point where probability drops below threshold
        trusted_indices = (candidates_prob > threshold).nonzero()
        if len(trusted_indices) == 0:
            return 1  # Default to at least one head
        
        # Get the longest consecutive streak of high-probability tokens
        consecutive_indices = []
        current_run = [trusted_indices[0].item()]
        
        for i in range(1, len(trusted_indices)):
            if trusted_indices[i].item() == trusted_indices[i-1].item() + 1:
                current_run.append(trusted_indices[i].item())
            else:
                # End of current run
                if len(current_run) > len(consecutive_indices):
                    consecutive_indices = current_run
                current_run = [trusted_indices[i].item()]
        
        # Check the last run
        if len(current_run) > len(consecutive_indices):
            consecutive_indices = current_run
            
        # If we found a consecutive run, use its length
        if consecutive_indices:
            prefix_length = min(len(consecutive_indices), max_heads)
        else:
            # Fallback to the highest probability token
            prefix_length = 1
            
        return prefix_length
    
    def tree_decoding_adaptive(
        self,
        tree_candidates,
        past_key_values,
        medusa_position_ids,
        input_ids,
        retrieve_indices,
        num_active_heads=None
    ):
        """
        Modified tree_decoding to handle variable number of heads.
        
        Args:
            tree_candidates: Input tokens for tree decoding
            past_key_values: Past key-value states for attention
            medusa_position_ids: Position IDs for the Medusa tree
            input_ids: Original input token IDs
            retrieve_indices: Indices for retrieving from the tree
            num_active_heads: Number of active heads to use (None means use all)
            
        Returns:
            medusa_logits, logits, outputs: Outputs from tree decoding
        """
        # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence
        position_ids = medusa_position_ids + input_ids.shape[1]
        
        # If num_active_heads is specified, limit the active positions
        if num_active_heads is not None:
            # Limit position_ids to only active heads
            active_position_ids = position_ids[:num_active_heads+1]  # +1 for base model
            
            # Create a limited tree_candidates tensor for active heads
            active_tree_candidates = tree_candidates[:, :num_active_heads+1]  # +1 for base model
            
            # Use the model to decode with limited tree
            tree_medusa_logits, outputs, tree_logits = self(
                active_tree_candidates,
                output_orig=True,
                past_key_values=past_key_values,
                position_ids=active_position_ids,
                medusa_forward=True,
            )
        else:
            # Standard tree decoding with all heads
            tree_medusa_logits, outputs, tree_logits = self(
                tree_candidates,
                output_orig=True,
                past_key_values=past_key_values,
                position_ids=position_ids,
                medusa_forward=True,
            )
        
        # Reorder the obtained logits based on the retrieve_indices
        logits = tree_logits[0, retrieve_indices]
        medusa_logits = tree_medusa_logits[:, 0, retrieve_indices]
        
        return medusa_logits, logits, outputs
    
    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=None,
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling = 'typical', 
        fast = True,
        adaptive_mode = None,
        entropy_threshold = 0.5
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
            adaptive_mode (str, optional): Defines the adaptive mode for determining the heuristic for number of Medusa heads.
            entropy_threshold (float, optional): Threshold for token probability in current step mode. Default is 0.5.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        prev_accept_length = 5

        for idx in range(max_steps):
            if adaptive_mode == "previous":
                num_heads = self.adaptive_based_on_previous_step(prev_accept_length)
        
                # Generate candidates using limited number of heads
                candidates, tree_candidates = generate_candidates(
                    medusa_logits[:num_heads],  # Use only the first num_heads
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                    temperature=temperature,
                    posterior_alpha=posterior_alpha,
                    posterior_threshold=posterior_threshold,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                )
                
                # Use tree decoding with limited number of heads
                medusa_logits, logits, outputs = self.tree_decoding_adaptive(
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                    num_active_heads=num_heads
                )

            elif adaptive_mode == "current":
                candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                    temperature=temperature,
                    posterior_alpha=posterior_alpha,
                    posterior_threshold=posterior_threshold,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                )
                
                # Calculate token probabilities
                if temperature > 0:
                    probs = torch.softmax(logits[:, :-1] / temperature, dim=-1)
                else:
                    probs = torch.softmax(logits[:, :-1], dim=-1)
                    
                candidates_prob = torch.gather(
                    probs, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
                ).squeeze(-1)
                
                # Determine optimal prefix length based on token probabilities
                num_heads = self.adaptive_based_on_current_step(
                    candidates_prob, 
                    threshold=entropy_threshold
                )
                
                # Use tree decoding with determined prefix length
                medusa_logits, logits, outputs = self.tree_decoding_adaptive(
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                    num_active_heads=num_heads
                )
                
            else:
                # Original non-adaptive Medusa
                candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                    temperature=temperature,
                    posterior_alpha=posterior_alpha,
                    posterior_threshold=posterior_threshold,
                    top_p=top_p,
                    sampling=sampling,
                    fast=fast,
                )
                
                medusa_logits, logits, outputs = tree_decoding(
                    self,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            prev_accept_length = accept_length + 1

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    pass

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass


class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
