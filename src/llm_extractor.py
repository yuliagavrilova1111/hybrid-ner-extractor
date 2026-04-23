import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class LLMExtractor:
    def __init__(self, model_name='Qwen/Qwen2.5-0.5B', lora_path='models/llm_lora'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=None,
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def extract(self, text: str) -> dict:
        prompt = f'Извлеки сущности из текста и верни результат в формате JSON.\nТекст: {text}<|endoftext|>'
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=None,
                min_new_tokens=1
            )
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            start = full_response.find('{')
            end = full_response.rfind('}')
            if start != -1 and start < end:
                res = full_response[start:end+1]
                return json.loads(res)
            else:
                return {}
        except json.JSONDecodeError:
            return {}

# Пример использования
if __name__ == '__main__':
    text = 'Пациент Иванов жалуется на кашель. Врач прописал Амоксициллин.'
    llm_extractor = LLMExtractor()
    response = llm_extractor.extract(text)
    print(response)