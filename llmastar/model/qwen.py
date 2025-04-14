import requests
import json
import os
import re
from .prompts.qwen_prompts import PARSE_QWEN, QWEN_PROMPTS

class Qwen:
    def __init__(self, variant="Qwen2.5-7B-Instruct", device=None, use_api=False, api_key=None, site_url=None, site_name=None):
        """
        Initialize the Qwen model.
        Args:
            variant (str): The variant of the Qwen model to use.
            Options:
                - "Qwen2.5-7B-Instruct"
                - "Qwen2.5-0.5B-Instruct"
                - "Qwen2.5-Math-7B"
            device (torch.device): The device to run the model on.
            use_api (bool): Whether to use the OpenRouter API instead of local inference.
            api_key (str): The API key for OpenRouter. If None, will try to get from environment.
            site_url (str): Site URL for rankings on OpenRouter.
            site_name (str): Site name for rankings on OpenRouter.
        Returns:
            None
        """
        self.use_api = use_api
        self.model_id = f"Qwen/{variant}"
        self.openrouter_model = f"qwen/{variant}"
        
        if use_api:
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("API key is required for API usage. Provide it directly or set OPENROUTER_API_KEY environment variable.")
            self.site_url = site_url
            self.site_name = site_name
        else:
            import torch
            import transformers
            
            if device is None:
                device = torch.device("cuda:0")
            
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device=device,
            )
            self.terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>")
            ]
    
    def ask(self, prompt):
        if self.use_api:
            return self._api_ask(prompt)
        else:
            return self._local_ask(prompt)
    
    def _local_ask(self, prompt):
        outputs = self.pipeline(
            prompt,
            max_new_tokens=8000,
            eos_token_id=self.terminators,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][len(prompt):]
        
    def _api_ask(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        
        # Parse the prompt to extract system and user parts
        messages = self._parse_prompt_to_messages(prompt)
            
        data = {
            "model": self.openrouter_model,
            "messages": messages
        }
        
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data)
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    
    def _parse_prompt_to_messages(self, prompt):
        """Parse the prompt to extract system and user parts for API format."""
        # Regular expressions to match Qwen format parts
        system_pattern = r'<\|im_start\|>system(.*?)<\|im_end\|>'
        user_pattern = r'<\|im_start\|>user(.*?)<\|im_end\|>'
        
        messages = []
        
        # Extract system message if present
        system_match = re.search(system_pattern, prompt, re.DOTALL)
        if system_match:
            system_content = system_match.group(1).strip()
            messages.append({"role": "system", "content": system_content})
        
        # Extract user message
        user_match = re.search(user_pattern, prompt, re.DOTALL)
        if user_match:
            user_content = user_match.group(1).strip()
            messages.append({"role": "user", "content": user_content})
        
        # If no structured format was found, use the entire prompt as user message
        if not messages:
            messages.append({"role": "user", "content": prompt})
            
        return messages
        
    def get_prompt(self, prompt_type, **kwargs):
        """Get a prompt template and format it with the provided parameters."""
        if prompt_type == "parse":
            return PARSE_QWEN.format(**kwargs)
        elif prompt_type in QWEN_PROMPTS:
            return QWEN_PROMPTS[prompt_type].format(**kwargs)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
