import torch
from transformers import AutoModel, AutoModelForCausalLM

# Load the model from Hugging Face
model_names = ["haoranxu/ALMA-7B", "Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.1", "Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.2", "Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.3", "Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.4", "Max-Bosch/ALMA-7B-pruned-GBLM-unstructured", "Max-Bosch/ALMA-7B-pruned-GBLM-unstructured-0.6"]

for model_name in model_names:
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Function to count non-zero parameters
    def count_nonzero_params(model):
        total_nonzero_params = 0
        for param in model.parameters():
            # Count only non-zero elements
            total_nonzero_params += (param != 0).sum().item()
        return total_nonzero_params

    # Compute the number of non-zero parameters and storage size
    nonzero_params = count_nonzero_params(model)

    # Convert non-zero parameters to bytes
    param_size_in_bytes = nonzero_params * 4  # Each parameter is typically a float32 (4 bytes)

    # Convert bytes to megabytes and gigabytes
    param_size_in_mb = param_size_in_bytes / (1024**2)
    param_size_in_gb = param_size_in_bytes / (1024**3)

    print(f"{model_name}: total number of non-zero parameters: {nonzero_params}")
    print(f"{model_name}: effective model size: {param_size_in_mb:.2f} MB ({param_size_in_gb:.2f} GB)")
