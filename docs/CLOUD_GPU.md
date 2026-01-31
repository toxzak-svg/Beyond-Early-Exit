# Running UTIO on Cloud GPUs

UTIO needs a CUDA GPU and enough VRAM for your model. This guide covers common cloud GPU options.

## Requirements

- **GPU**: NVIDIA (A100, H100, L40S, 4090, or similar)
- **VRAM**: ~14GB+ for Llama-2-7B; scale up for larger models
- **Software**: Python 3.10+, PyTorch with CUDA, vLLM

---

## Option 1: RunPod

1. **Create account**: [runpod.io](https://www.runpod.io/)
2. **Deploy a GPU pod**: Choose a template (e.g. "PyTorch") and a GPU (e.g. A100 40GB, L40S).
3. **SSH in** (or use the web terminal).
4. **Setup**:
   ```bash
   git clone https://github.com/YOUR_ORG/Beyond-Early-Exit.git
   cd Beyond-Early-Exit
   pip install -e .
   pip install vllm
   ```
5. **Run**:
   ```bash
   python examples/vllm_example.py
   # Or benchmarks:
   python benchmarks/benchmark_suite.py --model llama-7b --output benchmarks/results.json
   ```

---

## Option 2: Lambda Labs

1. **Create account**: [lambdalabs.com](https://lambdalabs.com/)
2. **Launch instance**: Pick a GPU instance (e.g. 1x A100, 1x H100).
3. **SSH in**: `ssh ubuntu@<instance-ip>`
4. **Setup and run** (same as RunPod steps 4–5 above).

---

## Option 3: Google Cloud (GCE with GPU)

1. **Create a VM** with a GPU:
   ```bash
   gcloud compute instances create utio-gpu \
     --zone=us-central1-a \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-a100,count=1 \
     --image-family=common-cu121 \
     --image-project=deeplearning-platform-release \
     --maintenance-policy=TERMINATE
   ```
2. **SSH**: `gcloud compute ssh utio-gpu --zone=us-central1-a`
3. **On the VM**:
   ```bash
   # CUDA/drivers usually pre-installed on DL image
   pip install torch vllm
   git clone https://github.com/YOUR_ORG/Beyond-Early-Exit.git && cd Beyond-Early-Exit
   pip install -e .
   python examples/vllm_example.py
   ```

---

## Option 4: AWS (EC2 GPU instance)

1. **Launch an instance**: Use a **Deep Learning AMI** (Ubuntu) or **Amazon Linux 2023** with an instance type that has GPUs (e.g. `g5.xlarge` for A10G, `p4d.24xlarge` for A100).
2. **SSH in** and install if needed:
   ```bash
   # DL AMI often has conda/pip; otherwise:
   pip install torch vllm
   git clone https://github.com/YOUR_ORG/Beyond-Early-Exit.git && cd Beyond-Early-Exit
   pip install -e .
   python examples/vllm_example.py
   ```

---

## Option 5: Vast.ai

1. **Create account**: [vast.ai](https://vast.ai/)
2. **Rent a GPU machine** (e.g. search for A100 or 4090).
3. **Connect** via SSH or Jupyter from the dashboard.
4. **Setup and run** (same as RunPod: clone repo, `pip install -e .`, `pip install vllm`, then run `examples/vllm_example.py` or benchmarks).

---

## Quick checklist (any provider)

| Step | Command / action |
|------|------------------|
| Clone | `git clone <repo> && cd Beyond-Early-Exit` |
| Install UTIO | `pip install -e .` |
| Install vLLM | `pip install vllm` |
| Verify GPU | `python -c "import torch; print(torch.cuda.is_available())"` |
| Run example | `python examples/vllm_example.py` |
| Run benchmarks | `python benchmarks/benchmark_suite.py --output benchmarks/results.json` |

---

## Model and VRAM

- **Llama-2-7B**: ~14GB VRAM (single GPU).
- **Larger models**: Use `tensor_parallel_size` in vLLM for multi-GPU, or pick a cloud instance with more VRAM (e.g. A100 80GB, H100).

To use a different model in the example, change the `model` argument in `examples/vllm_example.py` (and ensure you have access/huggingface token if the model is gated).
