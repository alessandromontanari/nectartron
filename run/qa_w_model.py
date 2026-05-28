import torch

from config.training_config import (
    AVAILABLE_MODELS, MAX_SEQ_LENGTH
)
from unsloth import FastLanguageModel


def main():

    for ii, model in enumerate(AVAILABLE_MODELS):
        print(f"  - {ii}: {model}")
    print("We are looking into expanding the model options, considering the VRAM constraints...")
    model_name = input("Enter the index of the model to use: ").strip()
    if not model_name.isdigit() or int(model_name) >= len(AVAILABLE_MODELS):
        print("Invalid model index.")
        exit(1)
    model_name = AVAILABLE_MODELS[int(model_name)]
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./fine-tuned-model/nectartron-v0" + f"-qa-{model_name.split('unsloth/')[1]}",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = FastLanguageModel.for_inference(model)

    prompt = "<s>[INST] What trigger mode was used in run #635? [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, 
        max_length=MAX_SEQ_LENGTH,
        temperature=0.7,
        top_p=0.9,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()