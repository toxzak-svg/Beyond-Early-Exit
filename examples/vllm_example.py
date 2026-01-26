"""Example: Using UTIO with vLLM."""

"""
This example shows how to integrate UTIO into vLLM for adaptive inference.

Note: This requires vLLM to be installed:
    pip install vllm

The integration patches vLLM's model executor to add UTIO routing.
"""

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not installed. Install with: pip install vllm")

from utio.vllm_integration import patch_vllm_model_executor, create_vllm_utio_config


def main():
    """Example usage of UTIO with vLLM."""
    if not VLLM_AVAILABLE:
        print("Skipping example - vLLM not available")
        return

    # Initialize vLLM model
    print("Loading model with vLLM...")
    llm = LLM(
        model="meta-llama/Llama-2-7b-hf",  # Or your model path
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    # Configure UTIO
    utio_config = create_vllm_utio_config(
        bucket_cutoffs=(6, 16, 32),
        min_bucket=8,
        min_depth=6,
    )

    # Patch model executor with UTIO
    print("Enabling UTIO routing...")
    wrapper = patch_vllm_model_executor(
        llm.llm_engine.model_executor,
        use_utio=True,
        **utio_config,
    )

    if wrapper:
        print("✓ UTIO enabled")
    else:
        print("✗ UTIO not enabled")

    # Generate text
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    prompts = [
        "What is 2+2?",
        "Explain quantum computing in detail.",
        "Write a haiku about AI.",
    ]

    print("\nGenerating responses...")
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n[{i+1}] Prompt: {prompt}")
        print(f"    Response: {generated_text[:100]}...")


if __name__ == "__main__":
    main()
