"""
Sampling script for fine-tuned FOMC transcript model.

This script loads the fine-tuned LoRA model and runs inference
using system and user prompts.
"""

import os
import sys
import tinker
from tinker import types
from transformers import AutoTokenizer

# Add parent directory to path to import from finetune
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from finetune.tinker_finetune import model_name, lora_rank

# Configuration
checkpoint_info_file = "../finetune/checkpoint_info.txt"
output_dir = "output"  # Directory to save generated samples

# Inference hyperparameters
max_tokens = 2048
temperature = 0.7
num_samples = 5


def load_checkpoint_path(checkpoint_info_file: str) -> str:
    """
    Load the sampler weights checkpoint path from checkpoint_info.txt.
    
    Args:
        checkpoint_info_file: Path to checkpoint_info.txt
    
    Returns:
        The sampler weights checkpoint path
    """
    if not os.path.exists(checkpoint_info_file):
        raise FileNotFoundError(
            f"Checkpoint info file not found: {checkpoint_info_file}\n"
            "Please run training first or provide the checkpoint path manually."
        )
    
    sampler_path = None
    with open(checkpoint_info_file, "r") as f:
        for line in f:
            if "Path:" in line and "sampler_weights" in line:
                # Extract the path from the line
                sampler_path = line.split("Path:", 1)[1].strip()
                break
            elif "SAMPLER WEIGHTS CHECKPOINT" in line:
                # Next line should have the path
                next_line = f.readline()
                if "Path:" in next_line:
                    sampler_path = next_line.split("Path:", 1)[1].strip()
                    break
    
    if not sampler_path:
        raise ValueError(
            f"Could not find sampler weights checkpoint path in {checkpoint_info_file}\n"
            "Make sure you've completed training and the checkpoint was saved."
        )
    
    return sampler_path


def load_prompts(user_prompt_file: str, system_prompt_file: str) -> tuple[str, str]:
    """
    Load prompts from markdown files.
    
    Args:
        user_prompt_file: Path to USER_PROMPT.md
        system_prompt_file: Path to SYSTEM_PROMPT.md
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    
    Raises:
        FileNotFoundError: If either prompt file doesn't exist
        ValueError: If either prompt file is empty
    """
    # Check if files exist
    if not os.path.exists(system_prompt_file):
        raise FileNotFoundError(
            f"System prompt file not found: {system_prompt_file}\n"
            "Please create SYSTEM_PROMPT.md with the system prompt."
        )
    
    if not os.path.exists(user_prompt_file):
        raise FileNotFoundError(
            f"User prompt file not found: {user_prompt_file}\n"
            "Please create USER_PROMPT.md with the user prompt."
        )
    
    # Load and validate system prompt
    with open(system_prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    if not system_prompt:
        raise ValueError(
            f"System prompt file is empty: {system_prompt_file}\n"
            "Please add content to SYSTEM_PROMPT.md"
        )
    
    print(f"  ✓ Loaded system prompt from {system_prompt_file} ({len(system_prompt)} characters)")
    
    # Load and validate user prompt
    with open(user_prompt_file, "r", encoding="utf-8") as f:
        user_prompt = f.read().strip()
    
    if not user_prompt:
        raise ValueError(
            f"User prompt file is empty: {user_prompt_file}\n"
            "Please add content to USER_PROMPT.md"
        )
    
    print(f"  ✓ Loaded user prompt from {user_prompt_file} ({len(user_prompt)} characters)")
    
    return system_prompt, user_prompt


def format_prompt_for_qwen(system_prompt: str, user_prompt: str, tokenizer) -> types.ModelInput:
    """
    Format system and user prompts for Qwen model.
    
    Qwen3 models use a specific chat template format.
    Uses the tokenizer's apply_chat_template if available, otherwise uses manual formatting.
    """
    # Try to use the tokenizer's chat template if available
    try:
        # Qwen3 tokenizers typically have a chat template
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to manual formatting
            formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    except Exception as e:
        # Fallback to manual formatting if chat template fails
        print(f"  Warning: Could not use chat template ({e}), using manual formatting")
        formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize the prompt
    tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
    
    # Create ModelInput
    return types.ModelInput.from_ints(tokens=tokens)


def sample_from_model(
    checkpoint_path: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    num_samples: int,
    output_dir: str = output_dir
):
    """
    Load the fine-tuned model and run inference.
    
    Args:
        checkpoint_path: Tinker checkpoint path for sampler weights
        system_prompt: System prompt text
        user_prompt: User prompt text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_samples: Number of samples to generate
    """
    print("=" * 60)
    print("FOMC Transcript Model Inference")
    print("=" * 60)
    
    # Initialize service client
    print("\nInitializing Tinker service client...")
    service_client = tinker.ServiceClient()
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
    
    # Create sampling client from checkpoint
    print(f"\nLoading fine-tuned model from checkpoint...")
    print(f"  Checkpoint path: {checkpoint_path}")
    sampling_client = service_client.create_sampling_client(model_path=checkpoint_path)
    print("✓ Model loaded successfully!")
    
    # Format the prompt
    print("\nFormatting prompt...")
    print(f"  System prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"  System prompt: {system_prompt}")
    print(f"  User prompt: {user_prompt[:100]}..." if len(user_prompt) > 100 else f"  User prompt: {user_prompt}")
    
    model_input = format_prompt_for_qwen(system_prompt, user_prompt, tokenizer)
    
    # Set up sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>", "\n\n\n"]  # Stop at end tokens or multiple newlines
    )
    
    # Run inference
    print(f"\nGenerating response (max_tokens={max_tokens}, temperature={temperature})...")
    future = sampling_client.sample(
        prompt=model_input,
        sampling_params=sampling_params,
        num_samples=num_samples
    )
    result = future.result()
    
    # Display results
    print("\n" + "=" * 60)
    print("Generated Response(s)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each sample to a file
    saved_files = []
    for i, seq in enumerate(result.sequences, 1):
        response_text = tokenizer.decode(seq.tokens)
        print(f"\nSample {i}:")
        print("-" * 60)
        print(response_text)
        print("-" * 60)
        
        # Save to file
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_{i:03d}_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Generated Sample {i}\n\n")
            f.write(f"**Generated at:** {datetime.now().isoformat()}\n")
            f.write(f"**Temperature:** {temperature}\n")
            f.write(f"**Max Tokens:** {max_tokens}\n\n")
            f.write("---\n\n")
            f.write(response_text)
        
        saved_files.append(filepath)
        print(f"  ✓ Saved to: {filepath}")
    
    print(f"\n✓ All {len(saved_files)} samples saved to {os.path.abspath(output_dir)}/")
    
    return result, saved_files


def main():
    """Main inference function."""
    # Get file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_info_path = os.path.join(script_dir, checkpoint_info_file)
    user_prompt_file = os.path.join(script_dir, "USER_PROMPT.md")
    system_prompt_file = os.path.join(script_dir, "SYSTEM_PROMPT.md")
    
    try:
        # Load checkpoint path
        checkpoint_path = load_checkpoint_path(checkpoint_info_path)
        
        # Load prompts
        system_prompt, user_prompt = load_prompts(user_prompt_file, system_prompt_file)
        
        # Resolve output directory path
        output_dir_path = os.path.join(script_dir, output_dir)
        
        # Run inference
        result, saved_files = sample_from_model(
            checkpoint_path=checkpoint_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
            output_dir=output_dir_path
        )
        
        print("\n" + "=" * 60)
        print("Inference completed!")
        print("=" * 60)
        print(f"\nSaved {len(saved_files)} sample(s) to {output_dir_path}/")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

