from calflops import calculate_flops
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from datasets import load_dataset

ds = load_dataset("haoranxu/ALMA-Human-Parallel", "de-en")

teacher_model_name = "haoranxu/ALMA-13B-R"

teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.float16, device_map="auto")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, padding_side='left')

student_model_name = "haoranxu/ALMA-7B-R"
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.float16, device_map="auto")
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, padding_side='left')

# Tokenization function
def tokenize_function(examples):
    return student_tokenizer(examples['translation'], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./distilled-model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    logging_dir='./logs',
    report_to="none",
)

# Distillation loss function (simplified version using teacher logits)
def compute_loss(model, inputs, teacher_logits):
    outputs = model(**inputs)
    student_logits = outputs.logits
    loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
    loss = loss_fn(student_logits.log_softmax(dim=-1), teacher_logits.softmax(dim=-1))
    return loss

# Custom training loop with teacher-student distillation
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Generate teacher logits
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits
        
        # Calculate student loss
        loss = compute_loss(model, inputs, teacher_logits)
        return (loss, teacher_outputs) if return_outputs else loss

# Initialize the trainer
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the student model
trainer.train()