#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, Optional

import torch
from transformers.generation import (
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint

from habitat_llm.llm.hf_model import HFModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class StopOnTokenSequence(StoppingCriteria):
    """
    Stopping criteria that stops generation when a specific token sequence is encountered.
    This is more robust than just setting eos_token_id for multi-token stop sequences.
    """

    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
        self.sequence_length = len(stop_token_ids)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Check if the last N tokens match the stop sequence
        if input_ids.shape[-1] >= self.sequence_length:
            last_tokens = input_ids[0, -self.sequence_length :].tolist()
            if last_tokens == self.stop_token_ids:
                return True
        return False


class Qwen(HFModel):
    """Load Qwen using Hugging Face (HF) or vLLM - simplified version based on Llama implementation"""

    def __init__(self, conf):
        # Check if using vLLM mode - handle separately before parent init
        self.llm_conf = conf
        self.generation_params = conf.generation_params
        self.max_tokens = self.generation_params.max_tokens
        self.inference_mode = conf.inference_mode

        if self.inference_mode == "vllm":
            self.init_vllm_client()
        else:
            super().__init__(conf)

    def init_vllm_client(self):
        """Initialize vLLM client using OpenAI-compatible API"""
        if OpenAI is None:
            raise ImportError("openai package is required for vLLM mode. Install with: pip install openai")

        # vLLM server endpoint - defaults to localhost:8000
        vllm_host = getattr(self.llm_conf, 'vllm_host', os.getenv('VLLM_HOST', 'localhost'))
        vllm_port = getattr(self.llm_conf, 'vllm_port', os.getenv('VLLM_PORT', '8000'))
        vllm_base_url = f"http://{vllm_host}:{vllm_port}/v1"

        self.vllm_client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require an API key
            base_url=vllm_base_url,
        )
        # Model name for vLLM (same as engine)
        self.vllm_model = self.generation_params.engine

    def generate(
        self,
        prompt,
        stop=None,
        max_length=None,
        generation_args=None,
        **kwargs,
    ):
        """Generate response using appropriate backend"""
        if self.inference_mode == "vllm":
            return self.generate_vllm(prompt, stop, max_length, generation_args)
        else:
            return super().generate(prompt, stop, max_length, generation_args, **kwargs)

    def generate_vllm(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        """Generate using vLLM's OpenAI-compatible API"""
        if stop is None:
            stop = self.generation_params.stop
        if max_length is None:
            max_length = self.generation_params.max_tokens

        # Convert stop to list format for OpenAI API
        stop_sequences = [stop] if isinstance(stop, str) else stop if stop else None

        try:
            response = self.vllm_client.completions.create(
                model=self.vllm_model,
                prompt=prompt,
                max_tokens=max_length,
                temperature=self.generation_params.temperature,
                top_p=self.generation_params.top_p,
                stop=stop_sequences,
                n=self.generation_params.n,
            )

            if self.generation_params.batch_response:
                self.batch_response = [choice.text.rstrip() for choice in response.choices]
                return self.batch_response
            else:
                self.response = response.choices[0].text.rstrip()
                return self.response

        except Exception as e:
            print(f"vLLM generation error: {e}")
            raise

    def generate_hf_llm(
        self,
        prompt: str,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        """
        Generate an output using hf
        :param prompt: A string with the input to the language model.
        :param stop: A string that determines when to stop generation
        :max_length: The max number of tokens to generate
        """
        # Prepare the model input from prompt
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Set the generating parameters
        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        gen_cfg.max_new_tokens = max_length
        gen_cfg.do_sample = self.generation_params.do_sample
        gen_cfg.num_return_sequences = self.generation_params.n
        gen_cfg.num_beams = self.generation_params.best_of
        gen_cfg.temperature = self.generation_params.temperature
        gen_cfg.repetition_penalty = self.generation_params.repetition_penalty
        gen_cfg.top_k = self.generation_params.top_k
        gen_cfg.top_p = self.generation_params.top_p
        gen_cfg.output_scores = True
        gen_cfg.return_dict_in_generate = True

        if stop is None:
            stop = self.generation_params.stop
        if max_length is None:
            max_length = self.generation_params.max_tokens

        # Set up stopping criteria for generation
        stopping_criteria = StoppingCriteriaList()

        if stop:
            if isinstance(stop, str):
                stop_token_ids = self.tokenizer.encode(stop, add_special_tokens=False)
                if stop_token_ids:
                    # Use sequence-based stopping criteria for multi-token stop sequences
                    stopping_criteria.append(StopOnTokenSequence(stop_token_ids))
                    # Also set the last token as eos for redundancy
                    gen_cfg.eos_token_id = stop_token_ids[-1]
            else:
                # Handle list of stop tokens - add criteria for each
                for s in stop:
                    stop_token_ids = self.tokenizer.encode(s, add_special_tokens=False)
                    if stop_token_ids:
                        stopping_criteria.append(StopOnTokenSequence(stop_token_ids))
                        # Set the last one as eos_token_id
                        gen_cfg.eos_token_id = stop_token_ids[-1]

        extra_generation_args = {}
        if generation_args is not None and "grammar_definition" in generation_args:
            # These add constrained generation to the hugging face
            # inference as a logits processor.
            # IncrementalGrammarConstraint contains the core code for running
            # the grammar based constraint
            # and GrammarConstrainedLogitsProcessor is mainly an interface
            # between Hugging Face and the grammar constraint.
            grammar = IncrementalGrammarConstraint(
                generation_args["grammar_definition"], "root", self.tokenizer
            )
            grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
            extra_generation_args["logits_processor"] = [grammar_processor]

        # Generate the response with stopping criteria
        self.response_raw = self.model.generate(
            **model_inputs,
            generation_config=gen_cfg,
            pad_token_id=self.tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria if len(stopping_criteria) > 0 else None,
            **extra_generation_args
        )

        # Only decode the response, not including the prompt
        input_prompt_len = model_inputs.input_ids.shape[1]
        # Decode the response
        # For Qwen models, explicitly set clean_up_tokenization_spaces to handle proper spacing
        decode_text = self.tokenizer.batch_decode(
            self.response_raw["sequences"][:, input_prompt_len:],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        # Process the response with improved stop sequence handling
        if self.generation_params.batch_response:
            # Return a list of response
            if isinstance(stop, str):
                self.batch_response = [
                    self._clean_response(res, stop) for res in decode_text
                ]
            else:
                self.batch_response = []
                for res in decode_text:
                    cleaned = res
                    for s in stop:
                        if s in cleaned:
                            cleaned = cleaned.split(s)[0]
                            break
                    self.batch_response.append(self._clean_response(cleaned, None))
        else:
            # Return a single response
            response = decode_text[0]
            if isinstance(stop, str):
                response = response.split(stop)[0]
            else:
                # Try each stop sequence
                for s in stop:
                    if s in response:
                        response = response.split(s)[0]
                        break
            self.response: str = self._clean_response(response, None)

    def _clean_response(self, text, stop=None):
        """
        Clean the response by removing stop sequences and special tokens.

        :param text: The text to clean
        :param stop: Optional stop sequence to remove
        :return: Cleaned text
        """
        # Remove stop sequence if present
        if stop and stop in text:
            text = text.split(stop)[0]

        # Remove common special tokens that might appear
        special_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        for token in special_tokens:
            text = text.replace(token, "")

        # Strip whitespace and return
        return text.rstrip()
