import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer
from tqdm import tqdm 

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
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    return parser

LANG_MAP = {
    'eng_Latn': 'English',
    'deu_Latn': 'German',
    'ces_Latn': 'Czech',
    'rus_Cyrl': 'Russian',
    'zho_Hans': 'Chinese'
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

def main():
    parser = get_parser()
    args = parser.parse_args()

    # set data dtype
    dtype_map = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}
    dtype = dtype_map.get(args.dtype, torch.float)

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    #model = PeftModel.from_pretrained(model, args.ckpt) # load when you have lora
    model.eval()
    tokenizer = LlamaTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

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

        # generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=args.beam, # beam size
                max_new_tokens=args.gen_max_tokens
            )

        
        outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Process and write the translations
        for prompt, output in zip(prompts, outputs):
            translation = output[len(prompt):].strip()
            file_out.write(translation.replace("\n", " ") + "\n")

    file_out.close()

if __name__ == "__main__":
    main()
