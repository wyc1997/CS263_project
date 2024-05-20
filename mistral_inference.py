from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,Dataset
import pandas

dataset = load_dataset("neural-bridge/rag-dataset-12000",split="test")

device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",device_map="auto", load_in_4bit=True) # load_in_8bit, or remove it for fp16
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

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
    messages = [
        {"role": "user", "content":message_raw},
    ]
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds#.to(device) # Note: not calling to(device) here causes a warning but no harm, but if i call to(device), it might throw an error when using multiGPU. Please double check
    # model.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    raw_output = decoded[0].split("[/INST]")[-1] # extract responses only
    raw_output = raw_output[:-4] # remove EOS token
    # print(raw_output)
    outList.append([context, question, goldenResponse ,raw_output])

df = pandas.DataFrame.from_dict(outList)
df.columns = ["context", "question", "goldenResponse", "modelOutput"]
df.to_csv("./mistral_inference_nums_"+str(prompting_range_start)+"-"+str(prompting_range_end)+".csv")