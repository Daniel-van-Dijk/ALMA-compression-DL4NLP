import argparse
import torch
from torch import nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
# from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm 
import time

SparseSemiStructuredTensor._FORCE_CUTLASS = True
@torch.compile
def to_sparse_semi_structured_compiled(x):
    return to_sparse_semi_structured(x)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fin', required=True)
    parser.add_argument('--fout', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--src', required=True)
    parser.add_argument('--tgt', required=True)
    parser.add_argument('--dtype', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--beam', type=int, required=True)
    parser.add_argument('--gen_max_tokens', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
    return parser

LANG_MAP = {
    'en': 'English',
    'de': 'German',
    'cs': 'Czech',
    'is': 'Icelandic',
    'zh': 'Chinese',
    'ru': 'Russian',
}

def dynamic_batching(tokenizer, texts, batch_size, max_length):
    """
    dynamic padding up to the longest sequence in the batch.
    """
    batch = []
    batch_length = 0

    for text in texts:
        input_length = len(tokenizer.encode(text, truncation=True, max_length=max_length))
        if len(batch) > 0 and (batch_length + input_length > max_length or len(batch) == batch_size):
            yield batch
            batch = []
            batch_length = 0
        
        batch.append(text)
        batch_length = max(batch_length, input_length)

    if len(batch) > 0:
        yield batch

def print_memory_usage(prefix=""):
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{prefix}Allocated: {allocated / (1024 ** 2):.2f} MB, Reserved: {reserved / (1024 ** 2):.2f} MB")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # set data dtype
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype, torch.float)

    print('[0]: Loading model...')
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    #model = PeftModel.from_pretrained(model, args.ckpt) # load when you have lora
    #for fqn, module in model.named_modules():
    #    if isinstance(module, nn.Linear) and "layer" in fqn:
    #        print_memory_usage(f"Mod {fqn}:")
    #        module.weight = nn.Parameter(to_sparse_semi_structured_compiled(module.weight))
    print('[1]: Done loading model...')
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print('[2]: Done loading tokenizer...')

    src = LANG_MAP[args.src]
    tgt = LANG_MAP[args.tgt]

    file_out = open(args.fout, "w")

    # read data
    with open(args.fin, 'r') as f:
        lines = f.readlines()

    # generate
    total_batches = (len(lines) + args.batch_size - 1) // args.batch_size  # calculate the number of batches
    for batch in tqdm(dynamic_batching(tokenizer, lines, args.batch_size, args.gen_max_tokens), total=total_batches, desc="Processing Batches"):
        prompts = []
        for line in batch:
            line = line.strip()
            # prepend prompt
            prompt = f"Translate this from {src} to {tgt}:\n{src}: {line}\n{tgt}:"
            prompts.append(prompt)
        
        # Tokenize with truncation and dynamic padding up to the longest sequence in the batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, ).to('cuda')
        
        t0 = time.time()

        # generate
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=args.beam, # beam size
                max_new_tokens=args.gen_max_tokens
            )
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        input_len = inputs.input_ids.shape[-1]
        output_len = generated_ids.shape[-1]
        tokens = output_len
        tflops = args.batch_size*0.01323*tokens*args.beam
        time_used = tflops/312
        print(f"outlen={output_len}, time used est. {time_used*1000:.2f}ms, real. {(time.time()-t0)*1000:.2f}ms", flush=True)

        # Process and write the translations
        for prompt, output in zip(prompts, outputs):
            translation = output[len(prompt):].strip()
            file_out.write(translation.replace("\n", " ") + "\n")

    file_out.close()

if __name__ == "__main__":
    main()
