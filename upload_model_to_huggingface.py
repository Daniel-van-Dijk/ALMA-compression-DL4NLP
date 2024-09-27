from huggingface_hub import HfApi, HfFolder, upload_folder
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str)
    parser.add_argument('--local_model_folder', type=str)
    args = parser.parse_args()

    # Upload the entire folder to the Hugging Face Hub
    upload_folder(
        folder_path=args.local_model_folder,
        repo_id=args.repo_id,
        repo_type="model"  # Since it's a model repository
    )

    # model = AutoModelForCausalLM.from_pretrained("Max-Bosch/ALMA-7B-pruned-GBLM-4to8")
    # tokenizer = AutoTokenizer.from_pretrained("Max-Bosch/ALMA-7B-pruned-GBLM-4to8")
    # print(model)
    # print(tokenizer)
