from transformers import AutoModelForCausalLM, AutoTokenizer
# import rmm
# rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
# from rmm.allocators.torch import rmm_torch_allocator
import torch
# torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
device = "cuda" # the device to load the model onto

class MistralModel:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",device_map=0, load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

    def __call__(self,message_raw):
        # message_raw = ""
        messages = [
            {"role": "user", "content":message_raw},
        ]
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(device) # Note: not calling to(device) here causes a warning but no harm, but if i call to(device), it might throw an error when using multiGPU. Please double check
        # model.to(device)
        generated_ids = self.model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)
        raw_output = decoded[0].split("[/INST]")[-1] # extract responses only
        raw_output = raw_output[:-4]
        return raw_output

    def encode_text_to_tokens(self, text: str) -> list[int]:
        """
        Encodes a given text string to a sequence of tokens using the model's tokenizer.

        Args:
            text (str): The text to encode.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        return self.tokenizer.encode(text)[1:]

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