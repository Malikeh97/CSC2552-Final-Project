#!/usr/bin/env python3
"""
Generate N responses using Tulu model for prompts with both_disagreement_and_variance controversy type.
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

def load_model(model_checkpoint):
    """Load the Tulu model and tokenizer."""
    print(f"Loading model from {model_checkpoint}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Generate a single response from the model."""
    
    # Always use instruct format for all models
    # Try to use the tokenizer's chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            # Fallback to manual format
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        # Manual format for Tulu models
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    
    # Tokenize with proper truncation
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Debug: Check input length
    input_length = inputs['input_ids'].shape[1]
    if input_length > 1800:
        print(f"    ⚠️ Warning: Long input ({input_length} tokens), may cause issues")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Add these to prevent early stopping
            min_new_tokens=10,  # Force at least 10 tokens
        )
    
    # Decode the FULL output first
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Debug: Check output length
    output_length = len(outputs[0])
    new_tokens = output_length - input_length
    
    # Extract response - try multiple methods
    response = ""
    
    # Method 1: Remove the formatted prompt by string matching
    if full_output.startswith(formatted_prompt):
        response = full_output[len(formatted_prompt):].strip()
    # Method 2: Try to find assistant marker
    elif "<|assistant|>" in full_output:
        response = full_output.split("<|assistant|>")[-1].strip()
    # Method 3: Just use everything after the prompt
    else:
        # Find where the prompt ends in the output
        prompt_clean = prompt.strip()
        if prompt_clean in full_output:
            idx = full_output.find(prompt_clean) + len(prompt_clean)
            response = full_output[idx:].strip()
        else:
            response = full_output.strip()
    
    # Debug info
    if not response or len(response) < 10:
        print(f"    Short/empty response detected:")
        print(f"    Input tokens: {input_length}, Generated tokens: {new_tokens}")
        print(f"    Full output length: {len(full_output)} chars")
        print(f"    Formatted prompt length: {len(formatted_prompt)} chars")
        print(f"    Full output (first 300 chars): '{full_output[:300]}'")
        print(f"    Extracted response: '{response[:100]}'")
    
    return response


def process_prompts(input_csv, output_csv, model_checkpoint, n_responses=15, 
                   max_new_tokens=512, temperature=0.7, top_p=0.9):
    """Process prompts and generate responses."""
    
    # Load input CSV
    print(f"Loading input CSV from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Filter for both_disagreement_and_variance
    filtered_df = df[df['controversy_type'] == 'both_disagreement_and_variance'].copy()
    print(f"Found {len(filtered_df)} prompts with controversy_type='both_disagreement_and_variance'")
    
    # Sample 100 examples to accelerate processing
    if len(filtered_df) > 100:
        print(f"Sampling 100 examples from {len(filtered_df)} prompts...")
        filtered_df = filtered_df.sample(n=100, random_state=42).reset_index(drop=True)
        print(f"Using {len(filtered_df)} sampled prompts")
    else:
        print(f"Using all {len(filtered_df)} prompts (less than 100)")
    
    if len(filtered_df) == 0:
        print("No prompts found with the specified controversy type. Exiting.")
        return
    
    # Load model
    model, tokenizer = load_model(model_checkpoint)
    
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
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Check if response is empty or too short
                if not response or len(response.strip()) == 0:
                    print(f"  Response {response_idx + 1}/{n_responses} is EMPTY - retrying once...")
                    # Retry once
                    response = generate_response(
                        model, tokenizer, prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    if not response or len(response.strip()) == 0:
                        print(f"Response {response_idx + 1}/{n_responses} still EMPTY after retry")
                
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
        print(f"\nSaved prompt {len(existing_df)}/{total_prompts + len(existing_prompts) - len(existing_df)} to {output_csv}")
    
    print(f"\n✓ Processing complete! All responses saved to {output_csv}")
    print(f"Final output format: {len(existing_df)} rows, 2 columns (prompt, responses)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses using Tulu model for controversial prompts"
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
        help="Path or name of the Tulu model checkpoint"
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
        top_p=args.top_p
    )


if __name__ == "__main__":
    main()