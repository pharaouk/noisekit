import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import numpy as np
import re
from torch.nn import CosineSimilarity

cwd = os.getcwd()
sys.setrecursionlimit(150000)
torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(torch.get_num_threads())
# torch.set_default_device('cuda')



#Load Model and Tokenizer
def load_model_tok(model_path, device = "cpu"):
    print( "\033[91m\033[1m" + "\n*****LOADING: " + model_path +"*****" + "\033[0m\033[0m")
    base_model_id = model_path
    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map=device, trust_remote_code=True) #torch_dtype=torch.bfloat16, 
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, device_map=device, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def get_total_params(model):
    return sum(p.numel() for p in model.parameters())

def get_model_arch(model):
    model_str = str(model)
    return model_str

def layer_sort_key(name):
    parts = name.split(".")
    for part in parts:
        if part.startswith("layers"):
            match = re.search(r"_([0-9]+)", part)
            if match:
                layer_num = int(match.group(1))
                return layer_num
    return -1

def get_layer_info(model):
    layer_info = []
    for name, param in model.named_parameters():
        layer_info.append((name, param.dtype))
    layer_info.sort(key=lambda x: layer_sort_key(x[0]))
    return layer_info


#Run a cosine similarity between each layer and its counterpart in both models
def get_layer_similarity(model_1, model_2):
    #create a list of tuples to store the layer names and similarities
    layer_similarity = []
    #loop through the layer names and parameters of both models
    for (name_1, param_1), (name_2, param_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        #assert that the layer names are the same
        # param_2 = torch.randn_like(vec1)

        try:
            assert name_1 == name_2
            #flatten the parameters into vectors
            print(name_1)
            print(param_1.size())
            print(param_2.size())

            vec_1 = param_1.view(-1)
            vec_2 = param_2.view(-1)

            print(vec_1.size())
            print(vec_2.size())
            #compute the cosine similarity between the vectors
            sim = torch.nn.functional.cosine_similarity(vec_1, vec_2, dim=0)
            print(sim)
            cos = CosineSimilarity(dim=0, eps=1e-8)
            #compute the cosine similarity between the tensors
            sim = cos(vec_1, vec_2)

            print(sim)
            #append the layer name and similarity to the list
            layer_similarity.append((name_1, sim.item()))
        except:
            pass
    # layer_similarity.sort(key=lambda x: x[0])
    layer_similarity.sort(key=lambda x: layer_sort_key(x[0]))
    avg_similarity = sum(layer_similarity) / len(layer_similarity)

    return layer_similarity, avg_similarity

# Import the cosine similarity module

# def get_layer_similarity(model_1, model_2):
#     #create a list of tuples to store the layer names and similarities
#     layer_similarity = []
#     #loop through the layer names and parameters of both models
#     for (name_1, param_1), (name_2, param_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
#         #assert that the layer names are the same
#         try:
#             assert name_1 == name_2
#             #flatten the parameters into vectors
#             vec_1 = param_1.view(-1)
#             vec_2 = param_2.view(-1)
#             #create an instance of the cosine similarity module with eps=1e-8
#             cos = CosineSimilarity(dim=0, eps=1e-8)
#             #compute the cosine similarity between the tensors
#             sim = cos(vec_1, vec_2)
#             #append the layer name and similarity to the list
#             layer_similarity.append((name_1, sim.item()))
#         except:
#             pass
#     #sort the list by the layer names
#     layer_similarity.sort(key=lambda x: layer_sort_key(x[0]))
#     return layer_similarity

#layers will have different names
#layers will have different shapes


def save_model_info(model, file_name):
    arch = get_model_arch(model)
    total_params = get_total_params(model)
    layer_info = get_layer_info(model)
    with open(file_name, "w") as f:
        f.write("Architecture:\n")
        f.write(arch + "\n")
        f.write("Parameter count:\n")
        f.write(str(total_params) + "\n")
        f.write("Layer names and data types:\n")
        for name, dtype in layer_info:
            f.write(name + " " + str(dtype) + "\n")
