#!/usr/bin/env python3
"""
Generate N responses using various LLM models for prompts with highly_controversial type.
Supports: Tulu, Llama 3.1, Qwen2.5, Mistral, and other models.
Saves results incrementally to a CSV file with compact format (one row per prompt with list of responses).
"""

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from tqdm import tqdm
import sys
import json
import re

# Check for sentencepiece (required for some models like Mistral, Llama)
try:
    import sentencepiece
except ImportError:
    print("WARNING: sentencepiece not installed. Some models (Mistral, Llama) may not work.")
    print("Install with: pip install sentencepiece")
    sentencepiece = None


# Model-specific chat templates
MODEL_TEMPLATES = {
    'tulu': {
        'pattern': r'tulu',
        'format': lambda prompt: f"<|user|>\n{prompt}\n<|assistant|>\n",
        'stop_tokens': ['<|user|>', '<|endoftext|>']
    },
    'llama-3': {
        'pattern': r'llama-3|llama3',
        'format': lambda prompt: f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        'stop_tokens': ['<|eot_id|>', '<|end_of_text|>']
    },
    'qwen': {
        'pattern': r'qwen',
        'format': lambda prompt: f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        'stop_tokens': ['<|im_end|>', '<|endoftext|>']
    },
    'mistral': {
        'pattern': r'mistral',
        'format': lambda prompt: f"[INST] {prompt} [/INST]",
        'stop_tokens': ['</s>', '[INST]']
    },
    'generic': {
        'pattern': r'.*',
        'format': lambda prompt: prompt,  # Use tokenizer's chat template or plain prompt
        'stop_tokens': []
    }
}


def detect_model_type(model_checkpoint):
    """Detect model type from checkpoint name."""
    checkpoint_lower = model_checkpoint.lower()
    
    for model_type, config in MODEL_TEMPLATES.items():
        if model_type == 'generic':
            continue
        if re.search(config['pattern'], checkpoint_lower):
            print(f"Detected model type: {model_type}")
            return model_type
    
    print("Model type not recognized, using generic template")
    return 'generic'


def format_prompt(prompt, model_type, tokenizer):
    """Format prompt according to model type."""
    
    # First, try to use the tokenizer's chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            print(f"  Using tokenizer's chat template")
            return formatted_prompt
        except Exception as e:
            print(f"  Warning: Could not use tokenizer chat template ({e}), falling back to manual format")
    
    # Fall back to manual formatting based on model type
    template = MODEL_TEMPLATES.get(model_type, MODEL_TEMPLATES['generic'])
    formatted_prompt = template['format'](prompt)
    print(f"  Using manual {model_type} format")
    return formatted_prompt


def load_model(model_checkpoint):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_checkpoint}...")
    
    # Try loading tokenizer with different configurations
    tokenizer = None
    tokenizer_error = None
    
    # First try with fast tokenizer
    try:
        print("Attempting to load tokenizer (fast version)...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint,
            trust_remote_code=True,
            use_fast=True
        )
        print("✓ Fast tokenizer loaded successfully")
    except Exception as e:
        tokenizer_error = e
        print(f"⚠️ Fast tokenizer failed: {e}")
        
        # Try with slow tokenizer
        try:
            print("Attempting to load tokenizer (slow version)...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_checkpoint,
                trust_remote_code=True,
                use_fast=False
            )
            print("✓ Slow tokenizer loaded successfully")
        except Exception as e2:
            print(f"✗ Slow tokenizer also failed: {e2}")
            
            # Check if sentencepiece is the issue
            if "sentencepiece" in str(e).lower() or "sentencepiece" in str(e2).lower():
                print("\n" + "="*60)
                print("ERROR: sentencepiece is required but not installed!")
                print("Please install it with:")
                print("  pip install sentencepiece")
                print("  or")
                print("  pip install protobuf sentencepiece")
                print("="*60 + "\n")
                sys.exit(1)
            else:
                raise Exception(f"Failed to load tokenizer: {e2}")
    
    # Load model
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    print("✓ Model loaded successfully!")
    
    # Detect model type
    model_type = detect_model_type(model_checkpoint)
    
    return model, tokenizer, model_type


def extract_response(full_output, formatted_prompt, prompt, model_type):
    """Extract the actual response from the full model output."""
    
    response = ""
    
    # Method 1: Remove the formatted prompt by string matching
    if full_output.startswith(formatted_prompt):
        response = full_output[len(formatted_prompt):].strip()
    
    # Method 2: Try model-specific markers
    elif model_type == 'tulu' and "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>")[-1].strip()
    
    elif model_type == 'llama-3' and "<|start_header_id|>assistant<|end_header_id|>" in full_output:
        parts = full_output.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    elif model_type == 'qwen' and "<|im_start|>assistant" in full_output:
        parts = full_output.split("<|im_start|>assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    elif model_type == 'mistral' and "[/INST]" in full_output:
        parts = full_output.split("[/INST]")
        if len(parts) > 1:
            response = parts[-1].strip()
    
    # Method 3: Find where the prompt ends
    else:
        prompt_clean = prompt.strip()
        if prompt_clean in full_output:
            idx = full_output.find(prompt_clean) + len(prompt_clean)
            response = full_output[idx:].strip()
        else:
            response = full_output.strip()
    
    # Clean up stop tokens from response
    template = MODEL_TEMPLATES.get(model_type, MODEL_TEMPLATES['generic'])
    for stop_token in template['stop_tokens']:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()
    
    return response


def generate_response(model, tokenizer, prompt, model_type, max_new_tokens=512, 
                     temperature=0.7, top_p=0.9):
    """Generate a single response from the model."""
    
    # Format prompt according to model type
    formatted_prompt = format_prompt(prompt, model_type, tokenizer)
    
    # Tokenize with proper truncation
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Debug: Check input length
    input_length = inputs['input_ids'].shape[1]
    if input_length > 1800:
        print(f"    ⚠️ Warning: Long input ({input_length} tokens), may cause issues")
    
    # Prepare generation config
    gen_config = {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'do_sample': True,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'min_new_tokens': 10,  # Force at least 10 tokens
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_config)
    
    # Decode the FULL output first
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Debug: Check output length
    output_length = len(outputs[0])
    new_tokens = output_length - input_length
    
    # Extract response using model-specific logic
    response = extract_response(full_output, formatted_prompt, prompt, model_type)
    
    # Final cleanup - remove any remaining special tokens
    response = tokenizer.decode(
        tokenizer.encode(response, add_special_tokens=False), 
        skip_special_tokens=True
    ).strip()
    
    # Debug info for short/empty responses
    if not response or len(response) < 10:
        print(f"    Short/empty response detected:")
        print(f"    Input tokens: {input_length}, Generated tokens: {new_tokens}")
        print(f"    Full output length: {len(full_output)} chars")
        print(f"    Full output (first 300 chars): '{full_output[:300]}'")
        print(f"    Extracted response: '{response[:100] if response else 'EMPTY'}'")
    
    return response


def process_prompts(input_csv, output_csv, model_checkpoint, n_responses=15, 
                   max_new_tokens=512, temperature=0.7, top_p=0.9, sample_size=100):
    """Process prompts and generate responses."""
    
    # Load input CSV
    print(f"Loading input CSV from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter for highly_controversial
    filtered_df = df[df['controversy_type'] == 'quality_variance'].copy()
    print(f"Found {len(filtered_df)} prompts with controversy_type='quality_variance'")
    
    # Sample examples to accelerate processing
    if sample_size and len(filtered_df) > sample_size:
        print(f"Sampling {sample_size} examples from {len(filtered_df)} prompts...")
        filtered_df = filtered_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Using {len(filtered_df)} sampled prompts")
    else:
        print(f"Using all {len(filtered_df)} prompts")
    
    if len(filtered_df) == 0:
        print("No prompts found with the specified controversy type. Exiting.")
        return
    
    # Load model
    model, tokenizer, model_type = load_model(model_checkpoint)
    
    # Check if output file exists and load existing data
    output_exists = os.path.exists(output_csv)
    if output_exists:
        print(f"Output file {output_csv} already exists. Loading existing data...")
        existing_df = pd.read_csv(output_csv)
        # Convert JSON string back to list
        existing_df['responses'] = existing_df['responses'].apply(json.loads)
        existing_prompts = set(existing_df['prompt'].values)
        print(f"Found {len(existing_prompts)} prompts already processed.")
    else:
        print(f"Creating new output file: {output_csv}")
        existing_df = pd.DataFrame(columns=['prompt', 'responses'])
        existing_prompts = set()
    
    # Process each prompt
    total_prompts = len(filtered_df)
    
    print(f"\nGenerating {n_responses} responses for each of {total_prompts} prompts...")
    print(f"Total generations: {total_prompts * n_responses}\n")
    
    for prompt_idx, row in tqdm(filtered_df.iterrows(), total=total_prompts, desc="Processing prompts"):
        prompt = row['prompt']
        
        # Skip if already processed
        if prompt in existing_prompts:
            print(f"\nSkipping prompt (already processed): {prompt[:50]}...")
            continue
        
        # Generate N responses for this prompt
        responses = []
        print(f"\n\nGenerating {n_responses} responses for prompt: {prompt[:100]}...")
        for response_idx in range(n_responses):
            try:
                response = generate_response(
                    model, tokenizer, prompt, model_type,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Check if response is empty or too short
                if not response or len(response.strip()) == 0:
                    print(f"  Response {response_idx + 1}/{n_responses} is EMPTY - retrying once...")
                    # Retry once
                    response = generate_response(
                        model, tokenizer, prompt, model_type,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    if not response or len(response.strip()) == 0:
                        print(f"  Response {response_idx + 1}/{n_responses} still EMPTY after retry")
                
                responses.append(response)
                print(f"  ✓ Response {response_idx + 1}/{n_responses} generated (length: {len(response)} chars)")
                
            except Exception as e:
                print(f"\n ERROR generating response {response_idx} for prompt idx {prompt_idx}:")
                print(f"   Error type: {type(e).__name__}")
                print(f"   Error message: {e}")
                import traceback
                traceback.print_exc()
                responses.append("")  # Add empty string for failed generations
                continue
        
        # Add new result to existing data
        new_row = pd.DataFrame([{
            'prompt': prompt,
            'responses': responses
        }])
        existing_df = pd.concat([existing_df, new_row], ignore_index=True)
        existing_prompts.add(prompt)
        
        # Save the entire dataframe (with all accumulated results)
        df_to_save = existing_df.copy()
        df_to_save['responses'] = df_to_save['responses'].apply(json.dumps)
        df_to_save.to_csv(output_csv, index=False)
        print(f"\nSaved prompt {len(existing_df)}/{total_prompts} to {output_csv}")
    
    print(f"\n✓ Processing complete! All responses saved to {output_csv}")
    print(f"Final output format: {len(existing_df)} rows, 2 columns (prompt, responses)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using various LLM models for controversial prompts. "
                   "Supports: Tulu, Llama 3.1, Qwen2.5, Mistral, and other models."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file with 'prompt' and 'controversy_type' columns"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to output CSV file for saving responses"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path or name of the model checkpoint (e.g., 'meta-llama/Llama-3.1-8B-Instruct')"
    )
    parser.add_argument(
        "--n_responses",
        type=int,
        default=15,
        help="Number of responses to generate per prompt (default: 15)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p (default: 0.9)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="Number of prompts to sample (default: 100, set to 0 for all prompts)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file {args.input_csv} does not exist!")
        sys.exit(1)
    
    # Process prompts
    process_prompts(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model_checkpoint=args.model_checkpoint,
        n_responses=args.n_responses,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        sample_size=args.sample_size if args.sample_size > 0 else None
    )


if __name__ == "__main__":
    main()