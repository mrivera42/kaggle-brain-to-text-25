import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLM():

    def __init__(self, model_path):

        self.model_path = model_path
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            # quantization_config=bnb,
            device_map="auto"
        )
    
    def infer(self, input_text): 

        # tokenize
        input_ids = self.tokenizer(input_text,return_tensors='pt').to(self.device)

        # infer 
        outputs = self.model.generate(**input_ids)

        # decode 
        outputs_decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return outputs_decoded
    
if __name__  == "__main__": 

    prompt = """Given this list of phonemes, output what you think is the likely sentence. 
    Phonemes are the smallest unit of sound in a language that can distinguish one word from another. 
    The | means silence (or space) . These are sounds, not words or tokens. using ARPAbet phonemes.
    
    Phoneme sequence: ['Y', 'UW', ' | ', 'K', 'AE', 'N', 'T', ' | ', 'M', 'EY', 'K', ' | ', 'AH', ' | ', 'D', 'IH', 'S', 'IH', 'ZH', 'AH', 'N', ' | ', 'BLANK', 'BLANK', 'BLANK']
    Likely sentence:
    """

    llm = LLM('Qwen2.5-7B-Instruct')

    print(llm.infer(prompt))

