from transformers import AutoModelForCausalLM, AutoTokenizer
# import rmm
# rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
# from rmm.allocators.torch import rmm_torch_allocator
import torch
# torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
device = "cuda" # the device to load the model onto

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
CHAT_TEMPLATE_ID = "HuggingFaceH4/zephyr-7b-beta"
model_name = "clibrain/mamba-2.8b-chat-no_robots"
eos_token = "<|endoftext|>"

class MambaModel:
    def __init__(self):
        self.model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=torch.bfloat16)# change here for FP32 inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.eos_token = eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_ID).chat_template
    
    def __call__(self,message_raw):
        messages = [
            {"role": "user", "content":message_raw},
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt",add_generation_prompt=True)
        model_inputs = encodeds.to(device)
        generated_ids = self.model.generate(model_inputs, max_length=3000,#2000
            temperature=0.9,#0.9
            top_p=0.7,#0.7
            eos_token_id=self.tokenizer.eos_token_id)
        decoded = self.tokenizer.batch_decode(generated_ids)
        raw_output = decoded[0].split("<|assistant|>\n")[-1].replace(eos_token, "")
        return raw_output

    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)

    def decode_tokens(self, tokens: list[int]) -> str:
        """
        Decodes a sequence of tokens back into a text string using the model's tokenizer.

        Args:
            tokens (list[int]): The sequence of token IDs to decode.
            context_length (Optional[int], optional): An optional length specifying the number of tokens to decode. If not provided, decodes all tokens.

        Returns:
            str: The decoded text string.
        """
        return self.tokenizer.decode(tokens)