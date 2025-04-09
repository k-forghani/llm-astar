from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Mistral:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def ask(self, prompt, max_new_tokens=200):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False
        )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer[len(prompt):].strip()
