import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import BitsAndBytesConfig

from config import config


class Saiga:
    def __init__(self):
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config['llm_model_name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(config['llm_model_name'])
        self.generation_config = GenerationConfig.from_pretrained(config['llm_model_name'])
        self.generation_config.top_k = 1
        self.generation_config.top_p = 0
        self.max_new_tokens = 1024
        print(self.generation_config)
        print(self.model.config.max_position_embeddings)

    def chat(self, messages):
        prompt = '<|begin_of_text|>'

        for mes in messages:
            if mes['role'] not in ['user', 'assistant', 'system']:
                raise Exception(f'{mes["role"]} is not in ["user", "assistant", "system"]')

            prompt += f'<|start_header_id|>{mes["role"]}<|end_header_id|>\n{mes["content"]}<|eot_id|>'
        prompt += '<|start_header_id|>assistant<|end_header_id|>'

        data = self.tokenizer(prompt, max_length=4096, truncation=True, return_tensors="pt", add_special_tokens=False).to('cuda')
        output_ids = self.model.generate(**data, generation_config=self.generation_config)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        torch.cuda.empty_cache()

        return output
