
import os
import glob
import time
import pandas
from tqdm import tqdm
from models.mistral import MistralModel
model = MistralModel()
# from models.mamba import MambaModel
# model = MambaModel()

context_lengths = [200,1000,8000,10000,16000,32000]
depth_percents = [0,25,50,75,100]
haystack_dir = "./PaulGrahamEssays"
needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
question = "What is the best thing to do in San Francisco?"
final_context_length_buffer = 150

def generate_context(context_length, depth_percent):
    # Get your haystack dir files loaded into a string
    context = read_context_files()
    # Truncate the haystack dir essays to the context length you desire
    context = encode_and_trim(context, context_length)
    # Insert your random statement according to your depth percent
    context = insert_needle(context, depth_percent, context_length)
    return context

def get_context_length_in_tokens(context):
    return len(model.encode_text_to_tokens(context))

def read_context_files():
        context = ""
        max_context_length = max(context_lengths)
        base_dir = os.path.abspath(os.path.dirname(__file__))  # Package directory

        while get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(os.path.join(base_dir, haystack_dir, "*.txt")):
                with open(file, 'r') as f:
                    context += f.read()
        return context

def encode_and_trim(context, context_length):
    tokens = model.encode_text_to_tokens(context)
    if len(tokens) > context_length:
        context = model.decode_tokens(tokens)[:context_length]
    return context

def insert_needle(context, depth_percent, context_length):
    tokens_needle = model.encode_text_to_tokens(needle)
    tokens_context = model.encode_text_to_tokens(context)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= final_context_length_buffer

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = model.encode_text_to_tokens('.')
        
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = model.decode_tokens(tokens_new_context)
    return new_context

outList = []
for i in tqdm(context_lengths):# context length
    for j in tqdm(depth_percents):#depth percent
        context = generate_context(i,j)
        # print(context)
        # Go see if the model can answer the question to pull out your random fact
        message = "Context: " + context + "\n\n Based on the above context, answer the following question: " + question
        response = model(message)
        outList.append([i,j,needle,response])

df = pandas.DataFrame.from_dict(outList)
df.columns = ["context_length", "depth_percent", "needle", "modelOutput"]
df.to_csv("./mistralNIAH.csv")