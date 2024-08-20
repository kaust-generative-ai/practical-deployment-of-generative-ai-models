# Practical deployment of Generative AI models

Course covering practical aspects of deploying, optimizing, and monitoring Generative AI models. The course is divided into three modules: Deployment, Model Optimization, and Monitoring and Maintenance Deployments.

### Module 1: Deployment

Covers various strategies for deploying Generative AI models starting from local deployment of Generative AI models on a laptop or workstation, followed by on-premise server-based deployments, then edge deployments, before finishing with cloud-based deployments. Cover the pros and cons of each strategy and the factors to consider when choosing a deployment strategy.

#### Module 1.1: Local deployments

1. [LLaMA C++](https://github.com/ggerganov/llama.cpp): Enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware - locally and in the cloud.
2. [LlamaFile](https://github.com/Mozilla-Ocho/llamafile): Make open-source LLMs more accessible to both developers and end users. Combines [LLaMA C++](https://github.com/ggerganov/llama.cpp) with [Cosmopolitan Libc](https://github.com/jart/cosmopolitan) into one framework that collapses all the complexity of LLMs down to a single-file executable (called a "llamafile") that runs locally on most computers, with no installation.
3. [Ollama](https://ollama.com) ([GitHub](https://github.com/ollama/ollama)): Get up and running with Llama 3, Mistral, Gemma 2, and other large language models. Uses [LLaMA C++](https://github.com/ggerganov/llama.cpp) as the backend.
4. [Open WebUI](https://openwebui.com) ([GitHub](https://github.com/open-webui/open-webui)): Extensible, self-hosted interface for AI that adapts to your workflow, all while operating entirely offline.

Additional relevant material:

* [DeepLearning AI: Open Source Models with HuggingFace](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/)
* [DeepLearning AI: Building Generative AI Apps](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/)
* [Blog Post: Emerging UI/UX patterns for AI applications](https://uxdesign.cc/emerging-interaction-patterns-in-generative-ai-experiences-8c351bb3392a)
* [Latent Space Podcast: Tiny Model Revolution](https://www.latent.space/p/cogrev-tinystories)

#### Module 1.2: On-premise, server-based deployments

* [OpenLLM](https://github.com/bentoml/OpenLLM)
* [VLLM](https://github.com/vllm-project/vllm)
* [Cog](https://github.com/replicate/cog)
* [TGI](https://github.com/huggingface/text-generation-inference)
* [TEI](https://github.com/huggingface/text-embeddings-inference)
* [DeepLearning AI: Building Generative AI Apps](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/)
* [Open WebUI](https://github.com/open-webui/open-webui)
 
#### Module 1.3: Edge deployments

* [DeepLearning AI: Introduction to Device AI](https://www.deeplearning.ai/short-courses/introduction-to-on-device-ai/)
* [Machine Learning Compilation](https://llm.mlc.ai) ([GitHub](https://github.com/mlc-ai/mlc-llm))
* [Deploying LLMs in your Web Browser](https://github.com/mlc-ai/web-llm)
* [NVIDIA Orin SDK](https://developer.nvidia.com/blog/deploy-large-language-models-at-the-edge-with-nvidia-igx-orin-developer-kit/)
* [NVIDIA Holoscan SDK](https://developer.nvidia.com/holoscan-sdk) ([GitHub](https://github.com/nvidia-holoscan/holoscan-sdk))
* [NVIDIA Holohub](https://github.com/nvidia-holoscan/holohub)

#### Module 1.4: Cloud-based deployments

* [DeepLearning AI: Serverless LLM Apps using AWS Bedrock](https://www.deeplearning.ai/short-courses/serverless-llm-apps-amazon-bedrock/)
* [DeepLearning AI: Developing Generative AI Apps using Microsoft Semantic Kernel](https://www.deeplearning.ai/short-courses/microsoft-semantic-kernel/)
* [DeepLearning AI: Understanding and Applying Text Embeddings with Vertex AI](https://www.deeplearning.ai/short-courses/google-cloud-vertex-ai/)
* [DeepLearning AI: Pair Programming with LLMs](https://www.deeplearning.ai/short-courses/pair-programming-llm/)
  
### Module 2: Model Optimization

Cover techniques for optimizing Generative AI models for deployment, such as model pruning, quantization, and distillation. Cover the trade-offs between model size, speed, and performance.

* [DeepLearning AI: Quantization Fundamentals](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [DeepLearning AI: Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)
* [GGUF My Repo](https://huggingface.co/spaces/ggml-org/gguf-my-repo)
  
### Module 3: Monitoring and Maintenance

Cover the importance of monitoring the performance of deployed models and updating them as needed. Discuss potential issues that might arise during deployment and how to troubleshoot them.

* [DeepLearning AI: Evaluating and Debugging Generative AI Apps](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/)
* [DeepLearning AI: LLMOps](https://www.deeplearning.ai/short-courses/llmops/)
* [DeepLearning AI: Automated Testing for LLMOps](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)
