from vllm import LLM, SamplingParams
import torch

# Sample prompts.
prompts = [
    "What is OpenVINO?",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=64, ignore_eos=True)

# Create an LLM.
llm = LLM(model="mistralai/Mistral-7B-v0.1", dtype=torch.float32, device='cpu', enforce_eager=True, trust_remote_code=True, seed=42, max_model_len=1024)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
