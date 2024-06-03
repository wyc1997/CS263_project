from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset,Dataset
import pandas
import json
import tqdm

checkpoint_path = "checkpoint/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873/"

dataset = load_dataset("neural-bridge/rag-dataset-12000",split="test")

device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(checkpoint_path,device_map="auto", load_in_4bit=True) # load_in_8bit, or remove it for fp16
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# IMPORTANT: CHANGE HERE FOR CHECKPOINTING
prompting_range_start = 2147
prompting_range_end = len(dataset)

outList = []
for i in tqdm.tqdm(range(prompting_range_start,prompting_range_end)):
    context = dataset["context"][i]
    question = dataset["question"][i]
    goldenResponse = dataset["answer"][i]
    if context is None or question is None or goldenResponse is None:
        continue
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
    with open('./mistral_inference_num_{}-{}.jsonl'.format(prompting_range_start, prompting_range_end), 'a') as result_file:
        result_file.write(json.dumps({"context": context, "question": question, "goldenResponse": goldenResponse, "raw_output": raw_output}))
        result_file.write("\n")


# df = pandas.DataFrame.from_dict(outList)
# df.columns = ["context", "question", "goldenResponse", "modelOutput"]
# df.to_csv("./mistral_inference_nums_"+str(prompting_range_start)+"-"+str(prompting_range_end)+".csv")