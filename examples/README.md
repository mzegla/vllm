# vLLM OpenVINO integration testing

## Prepare environment

Install Docker and create `workspace` environment as a working directory.

```bash
mkdir workspace && cd workspace
```

### Build Docker image of vLLM with OpenVINO integrated

```bash 
git clone --branch openvino-model-executor https://github.com/ilya-lavrenov/vllm.git
cd vllm
<docker build command>

```

Once it successfully finishes you will have a `vllm:openvino` image. It can directly spawn a serving container with OpenAI API endpoint or you can work with it interactively via bash shell.

## Run As A Serving

// TO DO

## Run Interactively

The `vllm:openvino` image does contain any samples by default, but since you have a vLLM repository cloned you can mount it to a container and use the samles from vLLM repository from the inside of the running container.