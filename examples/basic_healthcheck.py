import torch
import os
import csv
import traceback

import sys
sys.path.append(".")
from tests.conftest import VllmRunner, HfRunner

### Healthcheck configuration ###

# Tested models map (modelling file, HF model)
models = {
    "bichaun.py": "baichuan-inc/Baichuan-7B",
    "bloom.py": "bigscience/bloom-560m",
    "chatglm.py": "THUDM/chatglm3-6b",
    "decilm.py": "Deci/DeciLM-7B",
    "deepseek.py": "deepseek-ai/deepseek-moe-16b-base",
    "falcon.py": "tiiuae/falcon-7b",
    "gemma.py": "google/gemma-7b", # gated model
    "gpt2.py": "openai-community/gpt2",
    "gpt_bigcode.py": "bigcode/starcoder", # gated model
    "gpt_j.py": "EleutherAI/gpt-j-6b",
    "gpt_neox.py": "stabilityai/stablelm-tuned-alpha-7b",
    "internlm2.py" : "internlm/internlm2-7b",
    "llama.py (1)": "meta-llama/Llama-2-7b-chat-hf", # gated model
    "llama.py (2)": "mistralai/Mistral-7B-v0.1",
    "mixtral.py": "mistralai/Mixtral-8x7B-v0.1",
    #"mixtral_quant.py": "TheBloke/Mixtral-8x7B-v0.1-GGUF", # ???
    "mpt.py" : "mosaicml/mpt-7b-chat",
    "olmo.py": "allenai/OLMo-7B",
    "opt.py": "facebook/opt-6.7b",
    "orion.py": "OrionStarAI/Orion-14B-Base",
    "phi.py": "microsoft/phi-2",
    "qwen.py": "Qwen/Qwen-7B",
    "qwen2.py": "Qwen/Qwen1.5-7B",
    "stablelm.py": "stabilityai/stablelm-3b-4e1t",
    "starcoder2.py": "bigcode/starcoder2-7b",
    # neuron/mistral.py ?
    # neuron/llama.py ?
}

# Sample prompts.
prompts = [
    "What is OpenVINO?",
]

### Healthcheck ###
from enum import Enum
class ModellingMode(Enum):
    VLLM = "0"
    OPTIMUM = "1"

results = {ModellingMode.VLLM: {model: {"modelling_file": modelling_file} for modelling_file, model in models.items()},
    ModellingMode.OPTIMUM: {model: {} for model in models.values()} 
}

for modelling_mode in ModellingMode:
    print(f"Running healthcheck for {modelling_mode}")

    # We may want to use different kind of switch as Optimum should be the default modelling mode for OV
    os.environ["VLLM_OPENVINO_OPTIMUM"] = modelling_mode.value
    print(f"VLLM_OPENVINO_OPTIMUM: {os.environ['VLLM_OPENVINO_OPTIMUM']}")
    
    for modelling_file, model in models.items():

        print("\n______________________________________________________________________________________")
        if modelling_mode is ModellingMode.VLLM:
            print(f"\nChecking modelling file: {modelling_file} with model: {model}")
        else:
            print(f"\nChecking model: {model}")

        result = {
            "generation_passed": False,
            "exact_match_with_hf": False,
            "error_message": None,
            "stacktrace": None
        }

        # Create an LLM.
        try:
            vllm_model = VllmRunner(model, max_model_len=1024, dtype="float", device="auto", swap_space=100)
        except Exception as e:
            print(f"Error occurred during LLM object creation: {e}")
            result["error_message"] = f"Error occurred during LLM object creation: {e}"
            result["stacktrace"] = traceback.format_exc()
            results[modelling_mode][model].update(result)
            continue
        
        try:
            vllm_outputs = vllm_model.generate_greedy(prompts, max_tokens=128)
        except Exception as e:
            print(f"Error occurred during generation: {e}")
            result["error_message"] = f"Error occurred during generation: {e}"
            result["stacktrace"] = traceback.format_exc()
            results[modelling_mode][model].update(result)
            continue

        result["generation_passed"] = True
        ###########################
        try:
            hf_model = HfRunner(model, dtype="float", device="auto")
            hf_outputs = hf_model.generate_greedy(prompts, max_tokens=128)
        except Exception as e:
            print(f"Error in HF segment: {e}")
            result["error_message"] = f"Error in HF segment: {e}"
            result["stacktrace"] = traceback.format_exc()
            results[modelling_mode][model].update(result)
            continue
        #############################
        hf_output_ids, hf_output_str = hf_outputs[0]
        vllm_output_ids, vllm_output_str = vllm_outputs[0]
        print(f"Test string:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        print(f"Test ids:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")
        
        print(f"vLLM and HF outputs exact match: {hf_output_str == vllm_output_str and hf_output_ids == vllm_output_ids}")
        result["exact_match_with_hf"] = (hf_output_str == vllm_output_str and hf_output_ids == vllm_output_ids)
        results[modelling_mode][model].update(result)

print("\n\n##########################################################")
print("\nSUMMARY:\n")
for modelling_mode, modelling_results in results.items():
    print(f"\nResults for {modelling_mode}:")
    for model_name, result in modelling_results.items():
        print(f"\n{model_name}:")
        print(f"Generation passed: {result['generation_passed']}")
        if result["generation_passed"]:
            print(f"Exact match with HuggingFace output: {result['exact_match_with_hf']}")
        if result["error_message"]:
            print(f"Error message: {result['error_message']}")

with open("openvino_optimum_modelling.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["model", "generation passed", "exact output match", "error message"])
    for model_name, result in results[ModellingMode.OPTIMUM].items():
        writer.writerow([model_name, 
                         result["generation_passed"],
                         result["exact_match_with_hf"],
                         result["error_message"]])

with open("openvino_vllm_modelling.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerow(["modelling file", "model", "generation passed", "exact output match", "error message"])
    for model_name, result in results[ModellingMode.VLLM].items():
        writer.writerow([result["modelling_file"],
                         model_name, 
                         result["generation_passed"],
                         result["exact_match_with_hf"],
                         result["error_message"]])

with open("errors.txt", "w") as f:
    for modelling_mode in ModellingMode:
        f.write(f"\n\n####\nErrors captured in {modelling_mode}\n###")
        for model_name, result in results[modelling_mode].items():
            if result["stacktrace"] is not None:
                f.write(f"\n\n####\n{model_name}\n\n{result['error_message']}\n\n{result['stacktrace']}")
