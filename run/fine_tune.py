"""
Fine-tune a language model on NectarCAM Q&A dataset using Unsloth.
Optimized for NVIDIA RTX Pro 2000 Blackwell with limited VRAM.
"""

# TODO: implement a sort of second fine-tuning with another model 
# checking the answers of the first one and trying to correct them 
# (e.g. using a smaller model like tinyllama to check the answers of mistral-7b and correct them if they are wrong, then fine-tuning again with the corrected answers)
# reinforcement learning with human feedback (RLHF) approach, but using the smaller model as a sort of "critic" to improve the larger model's performance on the specific domain of NectarCAM Q&A

# TODO: mix the fine-tuning to use a dataset that is a mix of general knowledge 
# and specialized knowledge about the NectarCAM, to avoid overfitting and improve generalization

# TODO: https://unsloth.ai/docs/models/tutorials/ministral-3
# TODO: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Alpaca.ipynb#scrollTo=2ejIt2xSNKKp
# TODO: https://unsloth.ai/docs/get-started/unsloth-notebooks

from utils.environment import setup_environment, check_dependencies
from utils.dataset import load_qa_dataset, format_qa_for_training
from utils.model import load_model_and_tokenizer
from utils.training import train_model, save_model, test_inference
from config.training_config import (
    OUTPUT_DIR,
    DATASET_PATH,
    TEST_QUESTIONS,
)


def main():
    print("="*60)
    print("NectarTRON fine-tuning script")
    print("="*60 + "\n")
        
    # ============================================================================
    # SETUP & IMPORTS
    # ============================================================================
    setup_environment()
    check_dependencies()
    
    # ============================================================================
    # DATA LOADING & PROCESSING
    # ============================================================================
    qa_data = load_qa_dataset(DATASET_PATH)
    formatted_texts = format_qa_for_training(qa_data)
    
    # ============================================================================
    # MODEL SETUP
    # ============================================================================
    model, tokenizer, model_name = load_model_and_tokenizer()
    
    print(model_name.split("unsloth/")[1])

    # ============================================================================
    # TRAINING
    # ============================================================================
    model, tokenizer = train_model(
        model, 
        tokenizer, 
        formatted_texts, 
        OUTPUT_DIR + f"-qa-{model_name.split('unsloth/')[1]}"
    )
    
    # ============================================================================
    # SAVE & INFERENCE
    # ============================================================================
    save_model(
        model, 
        tokenizer, 
        OUTPUT_DIR + f"-qa-{model_name.split('unsloth/')[1]}"
    )

    test_inference(model, tokenizer, TEST_QUESTIONS)
    
    print(
        "\n✓ Fine-tuning complete! Model saved to:", 
        OUTPUT_DIR + f"-qa-{model_name.split('unsloth/')[1]}"
    )


if __name__ == "__main__":
    main()
