import transformers
import torch

class Qwen:
    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda:0")
        
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>")
        ]
    
    def ask(self, prompt):
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
