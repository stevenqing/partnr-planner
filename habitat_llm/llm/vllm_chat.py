#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""An LLM backend for an OpenAI-compatible server, which is what a local vLLM is.

The two backends this repository ships cannot reach one. `Llama` loads weights into the
evaluation process, which is impossible when forty of those processes run at once, and
`OpenAIChat` is hard-wired to Azure -- it builds `https://{OPENAI_ENDPOINT}` and offers no
way to say `http://127.0.0.1:8062/v1`. Everything else about the planners is unchanged;
this only replaces the transport, so a baseline run through it is PARTNR's own baseline.

Two details matter for comparing arms honestly. Sampling is greedy by default, so two arms
given the same prompt see the same decode. And a request that fails is retried and then
answered with an empty string rather than raising: a planner that receives nothing treats
it as an unparseable plan and moves on, which loses one step, whereas an exception loses
the whole episode and would quietly bias an arm's score by dropping its hard episodes.
"""

import os
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from habitat_llm.llm.base_llm import BaseLLM, Prompt


class VLLMChat(BaseLLM):
    """Chat-completions against any OpenAI-compatible endpoint."""

    def __init__(self, conf: DictConfig):
        from openai import OpenAI

        self.llm_conf = conf
        self.generation_params = self.llm_conf.generation_params
        base_url = os.getenv("VLLM_BASE_URL", "") or self.llm_conf.base_url
        api_key = os.getenv("VLLM_API_KEY", "") or self.llm_conf.get("api_key", "EMPTY")
        self.client = OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            max_retries=int(self.llm_conf.get("max_retries", 3)),
            timeout=float(self.llm_conf.get("request_timeout", 180)),
        )
        self.system_message = self.llm_conf.get("system_message", "")
        self.keep_message_history = bool(self.llm_conf.get("keep_message_history", False))
        self.verbose = bool(self.llm_conf.get("verbose", False))
        self.message_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[Any] = None,
        max_length: Optional[int] = None,
        generation_args: Any = None,
    ) -> str:
        """
        Generate a response for a prompt.

        :param prompt: a string, or a list of (kind, value) pairs for a multimodal turn.
        :param stop: stop sequence(s) overriding the configured ones.
        :param max_length: max tokens to generate, overriding the configured value.
        :param generation_args: grammar for constrained generation. Unsupported here --
            an OpenAI-compatible endpoint takes guided decoding through a vendor
            extension, and silently ignoring the grammar would let an arm configured for
            constrained generation look worse than it is, so this raises instead.
        """
        if generation_args:
            raise ValueError(
                "VLLMChat does not implement constrained generation; "
                "set plan_config.constrained_generation=False for this backend"
            )
        params = OmegaConf.to_object(self.generation_params)
        model = params.pop("model")
        params.pop("stream", None)
        if max_length is not None:
            params["max_tokens"] = max_length
        stop = stop if stop is not None else params.get("stop")
        params["stop"] = list(stop) if isinstance(stop, (list, tuple)) else stop

        messages: List[Dict[str, Any]] = list(self.message_history)
        if not messages and self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        else:
            content = [
                {"type": "text", "text": value}
                if kind == "text"
                else {"type": "image_url", "image_url": {"url": value, "detail": "low"}}
                for kind, value in prompt
            ]
            messages.append({"role": "user", "content": content})

        try:
            completion = self.client.chat.completions.create(
                model=model, messages=messages, **params
            )
            answer = completion.choices[0].message.content or ""
        except Exception as error:
            # See the note in the class docstring: an empty answer costs one planning
            # step, a raised exception costs the episode and biases the arm.
            print(f"[VLLMChat] request failed, answering empty: {type(error).__name__}: {error}")
            answer = ""

        if self.keep_message_history:
            self.message_history = messages + [{"role": "assistant", "content": answer}]
        if self.verbose:
            print(f"[VLLMChat] {answer}")
        return answer
