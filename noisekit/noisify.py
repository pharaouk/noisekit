
import argparse
import string
from noisekit.utils import get_layer_info, get_model_arch, get_total_params, load_model_tok
import yaml
import torch
import random
import argparse
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoModelForCausalLM, AutoTokenizer
import sys
from tqdm import tqdm

class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


def load_config(config_file):
    with open(config_file, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            config = Config(config_dict)
            return config
        except yaml.YAMLError as exc:
            print(exc)


            

def add_gradient_noise(model, noise_levels, sparsities, layer_names, layer_ranges, noise_types, param_types):
    assert len(noise_levels) == len(sparsities), "Noise levels and sparsities should have same length"
    
  
    for name, param in tqdm(model.named_parameters(), desc="Noisifying parameters", unit="parameter", file=sys.stdout):
        name_parts = name.split('.')
        layer_number = None
        for part in name_parts:
            if part.isdigit():
                layer_number = int(part)
                break

        if layer_number is None:
            layer_type = name_parts[1]
            layer_number = len(model.base_model.layers)
        else:
            layer_type = name_parts[name_parts.index(str(layer_number)) + 1]

        param_type = 'weight' if 'weight' in name else 'bias'




        if layer_type in layer_names and param_type in param_types and \
            (layer_number is None or any([start <= layer_number <= end for start, end in layer_ranges])):
            
            gradient_index = layer_number // (len(model.base_model.layers) // len(noise_levels))
            noise_level = noise_levels[gradient_index]
            sparsity = sparsities[gradient_index]

            noise_type = noise_types[layer_type]

            if noise_type == 'normal':
                noise = torch.randn(param.size()) * noise_level
            elif noise_type == 'uniform':
                noise = torch.rand(param.size()) * noise_level
            else:
                raise ValueError('Invalid noise type')
                
            mask = torch.rand(param.size()) < sparsity
            noise = noise * mask
            noise = noise.to(param.device)
            param.data.add_(noise)

def single_model_report(model):
    arch = get_model_arch(model)
    total_params = get_total_params(model)
    layer_info = get_layer_info(model)
    print( "\033[91m\033[1m" + "\n*****ARCHITECTURE*****" + "\033[0m\033[0m")
    print(arch)
    print( "\033[91m\033[1m" + "\n*****PARAM COUNT*****" + "\033[0m\033[0m")
    print(total_params)
    print( "\033[91m\033[1m" + "\n*****LAYER INFO*****" + "\033[0m\033[0m")
    print(layer_info)
    print( "\033[91m\033[1m" + "\n*****LAYER INFO 0*****" + "\033[0m\033[0m")
    layer_name, layer_dtype = layer_info[0]
    print(layer_name, layer_dtype)
    



def runner(output, config):
    base_model_id = config.base_model_id
    device = "cuda" if config.device == "cuda" else "cpu"
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map=device, torch_dtype=dtype)

    add_gradient_noise(
        model, 
        config.noise_levels, 
        config.sparsities, 
        config.layer_names, 
        config.layer_ranges, 
        config.noise_types, 
        config.param_types
    )

    model.save_pretrained(output, torch_dtype=torch.bfloat16)



def main():
    parser = argparse.ArgumentParser(description="Factory")
    parser.add_argument("--config", type=str, required=False, help="config path")
    parser.add_argument("--output", type=str, required=False, help="model name")
    

    args = parser.parse_args()

    if args.output:
        output = args.output

    else:
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        output = f"model_{random_suffix}"

    if args.config:
        config = load_config(args.config)

        runner(output, config)
    else:
        print("No Config has been provided")



def run_noisekit(config_path='config.yml'):
    config = load_config(config_path)
    if config.output:
        output = config.output
    else:
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        output = f"model_{random_suffix}"

    runner(output, config)

if __name__ == "__main__":
    main()





