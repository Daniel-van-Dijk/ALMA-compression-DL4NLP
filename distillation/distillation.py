from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
import torch
import argparse
from datasets import load_dataset, concatenate_datasets, DatasetDict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument('--distillation_weight', type=float, default=0.5)
    parser.add_argument('--output_dir', required=True)
    return parser

parser = get_parser()
args = parser.parse_args()

# Load teacher and student models
teacher_model_name = "haoranxu/ALMA-7B-R"
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16, device_map="auto")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, padding_side='left')

print("tm loaded")
student_model_name = args.backbone
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.bfloat16, device_map="auto")
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, padding_side='left')

print("sm loaded")
# Apply LoRA to the student model
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type: causal language modeling
    r=16,                           # Low-rank dimension
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Dropout rate for LoRA layers
    target_modules=["q_proj", "v_proj"]  # Apply LoRA to specific transformer modules (query, value projections)
)

print('Getting peft model', flush=True)
student_model = get_peft_model(student_model, lora_config)

print('Peft model got', flush=True)

# Tokenization function
def tokenize_function_to_en(examples):
    langlist = list(examples["translation"])
    src = langlist[1] if langlist[0]=="en"  else langlist[0]
    tgt = "en"
    line = examples["translation"][src]
    tline = examples["translation"][tgt]
    prompt = f"Translate this from {src} to {tgt}:\n{src}: {line}\n{tgt}: {tline}"
    # print(prompt)
    return student_tokenizer(prompt, truncation=True, padding="max_length", max_length=128)

def tokenize_function_from_en(examples):
    langlist = list(examples["translation"])
    tgt = langlist[1] if langlist[0]=="en"  else langlist[0]
    src = "en"
    line = examples["translation"][src]
    tline = examples["translation"][tgt]
    prompt = f"Translate this from {src} to {tgt}:\n{src}: {line}\n{tgt}: {tline}"
    # print(prompt)
    return student_tokenizer(prompt, truncation=True, padding="max_length", max_length=128)

# Load dataset
# List of all sub-datasets
sub_datasets = ["cs-en", "de-en", "is-en", "ru-en", "zh-en"]
# Load and concatenate all sub-datasets
ds_to_en = [
    load_dataset("haoranxu/ALMA-Human-Parallel", lang_pair)
    .map(tokenize_function_to_en, batched=False) 
    .remove_columns("translation")
for lang_pair in sub_datasets]
print(ds_to_en)
ds_from_en = [
    load_dataset("haoranxu/ALMA-Human-Parallel", lang_pair)
    .map(tokenize_function_from_en, batched=False) 
    .remove_columns("translation")
for lang_pair in sub_datasets]
print(ds_from_en)
tokenized_dataset = DatasetDict({
    'train': concatenate_datasets([d['train'] for d in ds_to_en]+[d['train'] for d in ds_from_en]),
    'validation': concatenate_datasets([d['validation'] for d in ds_to_en if 'validation' in d]+[d['validation'] for d in ds_from_en if 'validation' in d])
})
print("ds loaded")

# Training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    logging_dir='./logs',
    report_to="none",
)

# Loss function combining cross-entropy and KL divergence (teacher-student)
def compute_combined_loss(model, inputs, teacher_logits, distillation_weight=0.5):
    # print(inputs)
    # Forward pass for student model
    outputs = model(**inputs)
    student_logits = outputs.logits
    
    # Compute Cross-Entropy Loss (original loss)
    labels = inputs["input_ids"][:, 1:].clone()  
    labels = torch.cat([labels, torch.full((labels.size(0), 1), -100, dtype=torch.long).to(labels.device)], dim=1)
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ce_loss = ce_loss_fn(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    teacher_ce_loss = ce_loss_fn(teacher_logits.view(-1, teacher_logits.size(-1)), labels.view(-1))
    
    # Compute KL Divergence Loss (teacher-student loss)
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    kl_loss = kl_loss_fn(student_logits.log_softmax(dim=-1), teacher_logits.softmax(dim=-1))
    
    # Combine the two losses with a weight
    combined_loss = distillation_weight * kl_loss + (1 - distillation_weight) * ce_loss
    print(f'Computing Loss: student_ce_loss={ce_loss.item()}, teacher_ce_loss={teacher_ce_loss.item()}, kl_loss={kl_loss.item()}, combined_loss={combined_loss.item()}')
    return combined_loss

# Custom trainer with LoRA and distillation loss
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Generate teacher logits
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits
        
        # Calculate combined loss
        loss = compute_combined_loss(model, inputs, teacher_logits, distillation_weight=args.distillation_weight)
        return (loss, teacher_outputs) if return_outputs else loss

# Initialize the trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation']
)

print('Start Training', flush=True)

# Train the student model
trainer.train()
