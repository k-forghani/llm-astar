import requests
import json
import os
import re

class Mistral:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1", use_api=False, api_key=None, site_url=None, site_name=None):
        """
        Initialize the Mistral model.
        Args:
            model_name (str): The name/path of the Mistral model to use.
            use_api (bool): Whether to use the OpenRouter API instead of local inference.
            api_key (str): The API key for OpenRouter. If None, will try to get from environment.
            site_url (str): Site URL for rankings on OpenRouter.
            site_name (str): Site name for rankings on OpenRouter.
        Returns:
            None
        """
        self.use_api = use_api
        self.model_name = model_name
        self.openrouter_model = "mistralai/Mistral-7B-Instruct-v0.1"
        
        if use_api:
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
            if not self.api_key:
                raise ValueError("API key is required for API usage. Provide it directly or set OPENROUTER_API_KEY environment variable.")
            self.site_url = site_url
            self.site_name = site_name
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )

    def ask(self, prompt, max_new_tokens=200):
        if self.use_api:
            return self._api_ask(prompt)
        else:
            return self._local_ask(prompt, max_new_tokens)
            
    def _local_ask(self, prompt, max_new_tokens=200):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False
        )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer[len(prompt):].strip()
        
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
        """
        Parse the prompt to extract system and user parts for API format.
        For Mistral, we'll use a simple heuristic: 
        If the prompt contains '### System:' and '### User:', parse them accordingly.
        """
        system_pattern = r'### System:(.*?)(?=### User:|$)'
        user_pattern = r'### User:(.*?)(?=### Assistant:|$)'
        
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
