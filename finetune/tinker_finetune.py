
"""
Unsupervised LoRA fine-tuning script for Qwen 30B Instruct model
on FOMC press conference transcripts.

This script performs unsupervised language modeling fine-tuning to adapt
the model to generate text in the style of the FOMC transcripts.
"""

import os
import glob
import tinker
from tinker import types
from tinker_cookbook.hyperparam_utils import get_lora_lr_over_full_finetune_lr
from transformers import AutoTokenizer
import numpy as np
import urllib.request
from pathlib import Path

 
# Configuration
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
transcripts_dir = "../transcripts-txt"
lora_rank = 32
learning_rate = 1e-4
num_epochs = 10
batch_size = 4  # Number of examples per forward/backward pass
max_seq_length = 2048  # Maximum sequence length for training
gradient_accumulation_steps = 4  # Effective batch size = batch_size * gradient_accumulation_steps


def load_transcripts(transcripts_dir: str) -> list[str]:
    """Load all transcript files and return as a list of text strings."""
    transcript_files = glob.glob(os.path.join(transcripts_dir, "*.txt"))
    transcripts = []
    
    for file_path in sorted(transcript_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:  # Only add non-empty transcripts
                transcripts.append(text)
    
    print(f"Loaded {len(transcripts)} transcript files")
    return transcripts


def prepare_training_data(transcripts: list[str], tokenizer, max_seq_length: int) -> list[types.Datum]:
    """
    Prepare transcripts for unsupervised language modeling.
    
    For unsupervised fine-tuning, we treat the entire transcript as the target.
    All tokens have weight 1.0, meaning we want the model to learn to predict all tokens.
    """
    training_data = []
    total_tokens = 0
    
    for transcript in transcripts:
        # Tokenize the transcript
        tokens = tokenizer.encode(transcript, add_special_tokens=True)
        total_tokens += len(tokens)
        
        # Split into chunks if the transcript is too long
        for i in range(0, len(tokens), max_seq_length):
            chunk_tokens = tokens[i:i + max_seq_length]
            
            if len(chunk_tokens) < 2:  # Need at least 2 tokens for next-token prediction
                continue
            
            # For next-token prediction, input is tokens[:-1] and target is tokens[1:]
            input_tokens = chunk_tokens[:-1]
            target_tokens = chunk_tokens[1:]
            
            # All tokens have weight 1.0 for unsupervised learning
            weights = [1.0] * len(target_tokens)
            
            # Create the datum
            datum = types.Datum(
                model_input=types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": target_tokens,
                    "weights": weights
                }
            )
            
            training_data.append(datum)
    
    avg_tokens_per_transcript = total_tokens / len(transcripts) if transcripts else 0
    avg_tokens_per_example = total_tokens / len(training_data) if training_data else 0
    print(f"Prepared {len(training_data)} training examples from {len(transcripts)} transcripts")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Average tokens per transcript: {avg_tokens_per_transcript:.0f}")
    print(f"  - Average tokens per training example: {avg_tokens_per_example:.0f}")
    return training_data


def train_model(
    training_client,
    training_data: list[types.Datum],
    num_epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float
):
    """Perform the training loop."""
    # For gradient accumulation, we'll combine multiple batches
    effective_batch_size = batch_size * gradient_accumulation_steps
    num_batches = (len(training_data) + effective_batch_size - 1) // effective_batch_size
    total_steps = num_epochs * num_batches
    
    print(f"\nStarting training:")
    print(f"  - Total examples: {len(training_data)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Number of epochs: {num_epochs}")
    print(f"  - Total training steps: {total_steps}")
    print(f"  - Learning rate: {learning_rate}\n")
    
    step = 0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Shuffle data for each epoch
        np.random.shuffle(training_data)
        
        # Process in effective batch sizes (with gradient accumulation)
        for batch_idx in range(0, len(training_data), effective_batch_size):
            # Collect batches for gradient accumulation
            accumulated_batch = []
            for acc_idx in range(gradient_accumulation_steps):
                start_idx = batch_idx + (acc_idx * batch_size)
                end_idx = start_idx + batch_size
                if start_idx < len(training_data):
                    batch = training_data[start_idx:end_idx]
                    accumulated_batch.extend(batch)
            
            if not accumulated_batch:
                continue
            
            # Forward and backward pass on accumulated batch
            fwdbwd_future = training_client.forward_backward(
                accumulated_batch,
                "cross_entropy"
            )
            fwdbwd_result = fwdbwd_future.result()
            
            # Compute loss for monitoring
            # Following Tinker docs pattern exactly
            try:
                logprobs = np.concatenate([
                    output['logprobs'].tolist() 
                    for output in fwdbwd_result.loss_fn_outputs
                ])
                weights = np.concatenate([
                    example.loss_fn_inputs['weights'].tolist() 
                    for example in accumulated_batch
                ])
                
                # Ensure same length (in case of mismatch)
                min_len = min(len(logprobs), len(weights))
                logprobs = logprobs[:min_len]
                weights = weights[:min_len]
                
                # Compute weighted average log loss per token
                loss = -np.dot(logprobs, weights) / weights.sum()
            except Exception as e:
                # If there's an issue computing loss, just set to NaN
                print(f"Warning: Could not compute loss: {e}")
                loss = float('nan')
            
            # Optimizer step
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=learning_rate)
            )
            optim_result = optim_future.result()
            
            step += 1
            
            if step % 10 == 0 or step == total_steps:
                print(f"  Step {step}/{total_steps}, Loss: {loss:.4f}")


def extract_model_id_from_path(checkpoint_path: str) -> str:
    """
    Extract the model_id (training run ID) from a Tinker checkpoint path.
    
    Args:
        checkpoint_path: Tinker path (e.g., "tinker://<model_id>/<name>")
    
    Returns:
        The model_id (training run ID)
    """
    # Path format: tinker://<model_id>/<checkpoint_name>
    if checkpoint_path.startswith("tinker://"):
        parts = checkpoint_path.replace("tinker://", "").split("/")
        if len(parts) >= 1:
            return parts[0]
    return ""


def download_checkpoint(checkpoint_path: str, service_client, output_dir: str = "checkpoints"):
    """
    Download a checkpoint from Tinker to local storage.
    
    Args:
        checkpoint_path: Tinker path (e.g., "tinker://<model_id>/<name>")
        service_client: Tinker ServiceClient instance
        output_dir: Local directory to save the checkpoint
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create REST client to get download URL
        rest_client = service_client.create_rest_client()
        
        # Get the signed URL for downloading
        print(f"  Getting download URL for {checkpoint_path}...")
        future = rest_client.get_checkpoint_archive_url_from_tinker_path(checkpoint_path)
        checkpoint_archive_url_response = future.result()
        
        # Extract checkpoint name from path for filename
        checkpoint_name = checkpoint_path.split("/")[-1] if "/" in checkpoint_path else checkpoint_path.replace("tinker://", "")
        output_file = os.path.join(output_dir, f"{checkpoint_name}.tar")
        
        # Download the checkpoint
        print(f"  Downloading to {output_file}...")
        print(f"  URL expires at: {checkpoint_archive_url_response.expires}")
        urllib.request.urlretrieve(checkpoint_archive_url_response.url, output_file)
        
        print(f"✓ Checkpoint downloaded successfully!")
        print(f"  Local path: {os.path.abspath(output_file)}")
        print(f"  File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return output_file
    except Exception as e:
        print(f"✗ Error downloading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main training function."""
    print("=" * 60)
    print("FOMC Transcript Unsupervised LoRA Fine-tuning")
    print("=" * 60)
    
    # Get recommended learning rate
    lr_ratio = get_lora_lr_over_full_finetune_lr(model_name)
    print(f"LoRA LR ratio: {lr_ratio}")
    
    # Initialize service client
    service_client = tinker.ServiceClient()

    # Create training client
    print(f"\nCreating LoRA training client for {model_name}...")
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank
    )
    
    # Load tokenizer directly from model name (workaround for version compatibility)
    # This avoids the Qwen2Tokenizer import issue
    print(f"Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Ensure pad token is set (Qwen models may not have one by default)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"Error: Could not load tokenizer directly: {e}")
        print("\nThis error is likely due to an outdated transformers library.")
        print("Please try updating transformers:")
        print("  pip install --upgrade transformers")
        print("\nOr if that doesn't work, you may need to install the Qwen tokenizer:")
        print("  pip install transformers>=4.37.0")
        raise
    
    # Load transcripts
    transcripts_path = os.path.join(os.path.dirname(__file__), transcripts_dir)
    transcripts = load_transcripts(transcripts_path)
    
    if not transcripts:
        raise ValueError(f"No transcripts found in {transcripts_path}")
    
    # Prepare training data
    print("\nPreparing training data...")
    training_data = prepare_training_data(transcripts, tokenizer, max_seq_length)
    
    if not training_data:
        raise ValueError("No training data prepared")
    
    # Train the model
    train_model(
        training_client,
        training_data,
        num_epochs,
        batch_size,
        gradient_accumulation_steps,
        learning_rate
    )
    
    # Save the fine-tuned weights and optimizer state
    print("\n" + "=" * 60)
    print("Saving and Downloading Fine-tuned Weights")
    print("=" * 60)
    
    checkpoint_name = "fomc-transcripts-lora"
    
    try:
        # Save full state (weights + optimizer state) using save_state()
        # This allows full resumption of training if needed
        print(f"\nSaving checkpoint with full state: {checkpoint_name}...")
        print("  (This includes weights and optimizer state for resuming training)")
        save_state_result = training_client.save_state(name=checkpoint_name).result()
        full_state_path = save_state_result.path
        print("✓ Fine-tuned model state saved successfully!")
        print(f"  Full state checkpoint path: {full_state_path}")
        print(f"  Note: This checkpoint includes optimizer state and can be used to resume training")
        
        # Also save sampler weights for downloading
        # The download API only works with sampler weights checkpoints
        print(f"\nSaving sampler weights checkpoint for downloading...")
        sampler_save_result = training_client.save_weights_for_sampler(name=f"{checkpoint_name}-sampler").result()
        sampler_path = sampler_save_result.path
        print("✓ Sampler weights checkpoint saved!")
        print(f"  Sampler checkpoint path: {sampler_path}")
        
        # Extract and display model_id for future reference
        model_id = extract_model_id_from_path(full_state_path)
        print(f"  Model ID (training run ID): {model_id}")
        
        # Save checkpoint info to a file for easy reference
        checkpoint_info_file = "checkpoint_info.txt"
        with open(checkpoint_info_file, "w") as f:
            f.write(f"Checkpoint Name: {checkpoint_name}\n")
            f.write(f"Model ID: {model_id}\n\n")
            f.write(f"=== FULL STATE CHECKPOINT (for resuming training) ===\n")
            f.write(f"Path: {full_state_path}\n")
            f.write(f"Type: Weights + Optimizer State\n")
            f.write(f"Use for: Resuming training with load_state()\n")
            f.write(f"Download: NO (download API doesn't support full state)\n\n")
            f.write(f"=== SAMPLER WEIGHTS CHECKPOINT (for downloading) ===\n")
            f.write(f"Path: {sampler_path}\n")
            f.write(f"Type: Sampler Weights Only\n")
            f.write(f"Use for: Downloading and inference\n")
            f.write(f"Download: YES - Use this path for downloading!\n\n")
            f.write(f"To download via CLI:\n")
            f.write(f"  tinker checkpoint download {sampler_path}\n")
        print(f"  Checkpoint info saved to: {checkpoint_info_file}")
        print(f"\n  ⚠️  IMPORTANT: Use the SAMPLER path for downloading!")
        print(f"     Full state path: {full_state_path}")
        print(f"     Sampler path (for download): {sampler_path}")
        
        # Download the sampler weights (download API only works with sampler weights)
        print(f"\nDownloading checkpoint weights...")
        downloaded_file = download_checkpoint(sampler_path, service_client, output_dir="checkpoints")
        
        print("\n" + "=" * 60)
        print("Training and Download Completed Successfully!")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  ✓ Training completed")
        print(f"  ✓ Full state checkpoint saved: {full_state_path}")
        print(f"    (Use this to resume training with: training_client.load_state('{full_state_path}'))")
        print(f"  ✓ Sampler weights checkpoint saved: {sampler_path}")
        print(f"  ✓ Checkpoint downloaded: {downloaded_file}")
        print(f"  ✓ Checkpoint info saved: {checkpoint_info_file}")
        
    except Exception as e:
        print(f"\n✗ Error during save/download process: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Training completed, but checkpoint save/download failed.")
        print("You can try to download later using:")
        print("  python tinker_finetune.py download")
        raise


def download_only(checkpoint_path: str = None, checkpoint_name: str = "fomc-transcripts-lora", output_dir: str = "checkpoints"):
    """
    Download an existing checkpoint without training.
    
    Args:
        checkpoint_path: Full Tinker path (e.g., "tinker://<model_id>/<name>") - if provided, this is used directly
        checkpoint_name: Name of the checkpoint (used if checkpoint_path is not provided)
        output_dir: Local directory to save the checkpoint
    """
    print("=" * 60)
    print("Downloading FOMC Transcript LoRA Checkpoint")
    print("=" * 60)
    
    # Initialize service client
    service_client = tinker.ServiceClient()
    
    # Try to load checkpoint info from file first
    checkpoint_info_file = "checkpoint_info.txt"
    if checkpoint_path is None and os.path.exists(checkpoint_info_file):
        print(f"\nFound checkpoint info file: {checkpoint_info_file}")
        with open(checkpoint_info_file, "r") as f:
            for line in f:
                if line.startswith("Checkpoint Path:"):
                    checkpoint_path = line.split(":", 1)[1].strip()
                    print(f"  Loaded checkpoint path: {checkpoint_path}")
                    break
    
    # If checkpoint_path is still not provided, try to construct it
    if checkpoint_path is None:
        print(f"\nConstructing checkpoint path for: {checkpoint_name}")
        print("  (This requires the same model and rank configuration)")
        
        # Create a training client to get the model_id (same as used during training)
        training_client = service_client.create_lora_training_client(
            base_model=model_name,
            rank=lora_rank
        )
        
        # Get the model_id and construct the path
        model_id = training_client._guaranteed_model_id()
        checkpoint_path = f"tinker://{model_id}/{checkpoint_name}"
        print(f"  Constructed path: {checkpoint_path}")
        print(f"  Model ID: {model_id}")
    else:
        print(f"\nUsing checkpoint path: {checkpoint_path}")
        model_id = extract_model_id_from_path(checkpoint_path)
        if model_id:
            print(f"  Model ID: {model_id}")
    
    # Download the checkpoint
    try:
        download_checkpoint(checkpoint_path, service_client, output_dir)
        print("\n" + "=" * 60)
        print("Download completed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: Could not download checkpoint")
        print(f"  Path: {checkpoint_path}")
        print(f"  Error: {e}")
        print("\nHow to get the model_id:")
        print("  1. Check checkpoint_info.txt (created during training)")
        print("  2. Use the full checkpoint path from training output")
        print("  3. Use: python tinker_finetune.py download <full_path>")
        print("  4. Or list checkpoints using REST client:")
        print("     rest_client = service_client.create_rest_client()")
        print(f"     checkpoints = rest_client.list_checkpoints('{model_id}').result()")
        raise


def create_sampler_checkpoint_from_existing(checkpoint_name: str = "fomc-transcripts-lora"):
    """
    Create a sampler weights checkpoint from an existing training run.
    This is useful if you only saved the full state checkpoint and need to download.
    """
    print("=" * 60)
    print("Creating Sampler Weights Checkpoint for Download")
    print("=" * 60)
    
    service_client = tinker.ServiceClient()
    
    # Create a training client with the same configuration
    print(f"\nCreating training client with same config...")
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank
    )
    
    # Load the existing full state checkpoint if it exists
    # First, try to get the checkpoint path from the info file
    checkpoint_info_file = "checkpoint_info.txt"
    full_state_path = None
    
    if os.path.exists(checkpoint_info_file):
        with open(checkpoint_info_file, "r") as f:
            for line in f:
                if "Full State Path:" in line or "Checkpoint Path:" in line:
                    full_state_path = line.split(":", 1)[1].strip()
                    break
    
    if full_state_path:
        print(f"\nFound existing checkpoint: {full_state_path}")
        print("Loading checkpoint state...")
        try:
            training_client.load_state(full_state_path)
            print("✓ Checkpoint loaded successfully!")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint (this is okay if training is still active): {e}")
            print("   Continuing to create sampler checkpoint from current state...")
    
    # Now save the sampler weights checkpoint
    sampler_name = f"{checkpoint_name}-sampler"
    print(f"\nSaving sampler weights checkpoint: {sampler_name}...")
    sampler_result = training_client.save_weights_for_sampler(name=sampler_name).result()
    sampler_path = sampler_result.path
    
    print("✓ Sampler weights checkpoint created!")
    print(f"  Sampler checkpoint path: {sampler_path}")
    
    # Update checkpoint info file
    model_id = extract_model_id_from_path(sampler_path)
    with open(checkpoint_info_file, "a") as f:
        f.write(f"\n=== SAMPLER WEIGHTS CHECKPOINT (for downloading) ===\n")
        f.write(f"Path: {sampler_path}\n")
        f.write(f"Type: Sampler Weights Only\n")
        f.write(f"Use for: Downloading and inference\n")
        f.write(f"Download: YES - Use this path for downloading!\n\n")
        f.write(f"To download via CLI:\n")
        f.write(f"  tinker checkpoint download {sampler_path}\n")
    
    print(f"\n✓ Checkpoint info updated in: {checkpoint_info_file}")
    print(f"\nYou can now download using:")
    print(f"  tinker checkpoint download {sampler_path}")
    print(f"\nOr use the Python script:")
    print(f"  python tinker_finetune.py download {sampler_path}")
    
    return sampler_path


def get_checkpoint_info(checkpoint_name: str = "fomc-transcripts-lora"):
    """
    Get checkpoint information from an existing training run.
    This creates a new training client with the same config to access the checkpoint.
    """
    print("=" * 60)
    print("Retrieving Checkpoint Information")
    print("=" * 60)
    
    service_client = tinker.ServiceClient()
    
    # Create a training client with the same configuration
    print(f"\nCreating training client with same config as training...")
    print(f"  Base model: {model_name}")
    print(f"  LoRA rank: {lora_rank}")
    
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank
    )
    
    # Get the model_id (training run ID)
    model_id = training_client._guaranteed_model_id()
    print(f"\n✓ Model ID (Training Run ID): {model_id}")
    
    # Construct the checkpoint path
    checkpoint_path = f"tinker://{model_id}/{checkpoint_name}"
    print(f"✓ Checkpoint path: {checkpoint_path}")
    
    # Try to verify the checkpoint exists by attempting to get download URL
    print(f"\nVerifying checkpoint exists...")
    try:
        rest_client = service_client.create_rest_client()
        future = rest_client.get_checkpoint_archive_url_from_tinker_path(checkpoint_path)
        response = future.result()
        print(f"✓ Checkpoint verified! URL expires at: {response.expires}")
        
        # Save the info to file
        checkpoint_info_file = "checkpoint_info.txt"
        with open(checkpoint_info_file, "w") as f:
            f.write(f"Checkpoint Name: {checkpoint_name}\n")
            f.write(f"Checkpoint Path: {checkpoint_path}\n")
            f.write(f"Model ID: {model_id}\n")
        print(f"\n✓ Checkpoint info saved to: {checkpoint_info_file}")
        
        return checkpoint_path, model_id
    except Exception as e:
        print(f"✗ Could not verify checkpoint: {e}")
        print(f"\nThe checkpoint might:")
        print(f"  1. Have a different name (not '{checkpoint_name}')")
        print(f"  2. Have expired or been deleted")
        print(f"  3. Be from a different training run")
        print(f"\nTry listing all checkpoints for this model_id:")
        print(f"  rest_client = service_client.create_rest_client()")
        print(f"  checkpoints = rest_client.list_checkpoints('{model_id}').result()")
        raise


def list_checkpoints_for_model():
    """
    List all checkpoints for the current model configuration.
    """
    print("=" * 60)
    print("Listing All Checkpoints")
    print("=" * 60)
    
    service_client = tinker.ServiceClient()
    
    # Create a training client to get the model_id
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=lora_rank
    )
    
    model_id = training_client._guaranteed_model_id()
    print(f"\nModel ID: {model_id}")
    
    # List checkpoints
    try:
        rest_client = service_client.create_rest_client()
        print(f"\nFetching checkpoints...")
        response = rest_client.list_checkpoints(model_id).result()
        
        # CheckpointsListResponse is an object, need to access its attributes
        # Try common attribute names
        checkpoints_list = None
        if hasattr(response, 'checkpoints'):
            checkpoints_list = response.checkpoints
        elif hasattr(response, 'items'):
            checkpoints_list = response.items
        elif hasattr(response, 'data'):
            checkpoints_list = response.data
        elif hasattr(response, 'results'):
            checkpoints_list = response.results
        elif isinstance(response, (list, tuple)):
            checkpoints_list = response
        else:
            # Try to inspect the object
            print(f"\nResponse type: {type(response)}")
            print(f"Response attributes: {dir(response)}")
            # Try to access it as a list-like object
            try:
                checkpoints_list = list(response)
            except:
                checkpoints_list = [response]
        
        if checkpoints_list:
            print(f"\nFound {len(checkpoints_list)} checkpoint(s):")
            for i, ckpt in enumerate(checkpoints_list, 1):
                # Try to get a string representation
                if hasattr(ckpt, 'path'):
                    print(f"\n{i}. Path: {ckpt.path}")
                elif hasattr(ckpt, 'name'):
                    print(f"\n{i}. Name: {ckpt.name}")
                elif hasattr(ckpt, '__str__'):
                    print(f"\n{i}. {ckpt}")
                else:
                    print(f"\n{i}. {repr(ckpt)}")
        else:
            print(f"\nNo checkpoints found for model_id: {model_id}")
    except Exception as e:
        print(f"✗ Error listing checkpoints: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nNote: The model_id might be different if you used different")
        print(f"      hyperparameters (rank, base_model) during training.")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "download":
            # If a full path is provided (starts with "tinker://"), use it directly
            if len(sys.argv) > 2 and sys.argv[2].startswith("tinker://"):
                checkpoint_path = sys.argv[2]
                download_only(checkpoint_path=checkpoint_path)
            else:
                # Otherwise use as checkpoint name
                checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "fomc-transcripts-lora"
                download_only(checkpoint_name=checkpoint_name)
        
        elif command == "info":
            # Get checkpoint information
            checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "fomc-transcripts-lora"
            get_checkpoint_info(checkpoint_name)
        
        elif command == "create-sampler":
            # Create a sampler checkpoint from existing training
            checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "fomc-transcripts-lora"
            create_sampler_checkpoint_from_existing(checkpoint_name)
        
        elif command == "list":
            # List all checkpoints
            list_checkpoints_for_model()
        
        else:
            print("Usage:")
            print("  python tinker_finetune.py              # Run training")
            print("  python tinker_finetune.py download     # Download checkpoint")
            print("  python tinker_finetune.py info         # Get checkpoint info")
            print("  python tinker_finetune.py list         # List all checkpoints")
    else:
        main()

