from calflops import calculate_flops
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from torchprofile import profile_macs

model_name = "haoranxu/ALMA-7B-R"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

t0 = time.time()

for i in range(10):
    prompt="Translate this from English to German:\nEnglish: A quick brown fox jumps over the lazy dog.\nGerman:"
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, max_length=40, truncation=True).input_ids.cuda()
    macs = profile_macs(linear_model, sample_data)
    print(macs)
    with torch.no_grad():
        generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=20, do_sample=True, temperature=0.6, top_p=0.9)
    
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(outputs)

t1 = time.time()

execution_time = t1-t0  # Calculate the execution time
print(f"Execution time: {execution_time:.4f} seconds")