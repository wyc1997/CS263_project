from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from datasets import load_dataset,Dataset
import pandas
import torch

CHAT_TEMPLATE_ID = "HuggingFaceH4/zephyr-7b-beta"

model_name = "clibrain/mamba-2.8b-chat-no_robots"
# model_name = "clibrain/mamba-2.8b-instruct-openhermes"

dataset = load_dataset("neural-bridge/rag-dataset-12000",split="test")

eos_token = "<|endoftext|>"
device = "cuda" # note: mamba single GPU inference only
model = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=torch.float16)# change here for FP32 inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.eos_token = eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_ID).chat_template

# IMPORTANT: CHANGE HERE FOR CHECKPOINTING
prompting_range_start = 0
prompting_range_end = 2#len(dataset)

outList = []
for i in range(prompting_range_start,prompting_range_end):
    context = dataset["context"][i]
    question = dataset["question"][i]
    goldenResponse = dataset["answer"][i]
    message_raw = "Context: \n"+context + "\n given the above context, answer the question: "+question
    # print(message_raw)
    messages = []
    messages.append(dict(role="user", content=message_raw))
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt",add_generation_prompt=True)
    model_inputs = encodeds.to(device)
    # model.to(device)
    generated_ids = model.generate(model_inputs, max_length=2000,
        temperature=0.9,#0.9
        top_p=0.7,#0.7
        eos_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    raw_output = decoded[0].split("<|assistant|>\n")[-1].replace(eos_token, "")
    # print(raw_output)
    outList.append([context, question, goldenResponse ,raw_output])

df = pandas.DataFrame.from_dict(outList)
df.columns = ["context", "question", "goldenResponse", "modelOutput"]
df.to_csv("./mambaChat_inference_nums_"+str(prompting_range_start)+"-"+str(prompting_range_end)+".csv")