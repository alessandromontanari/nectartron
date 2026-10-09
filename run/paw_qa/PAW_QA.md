## Instructions

From root of the project:

Compile PAW program and save it.

```bash
python -m run.paq_qa.compile_paw_program
```
Train LoRA adapter for future QA. 

IMPORTANT: you will be prompted to choose for which PAW program you want to train the adapter. It must be for the same that you have just compiled.

```bash
python -m run.paq_qa.train_paw_lora_adapter
```

Convert LoRA adapter weights (? maybe they are not weights, anyways...) to GGUF.

```bash
cd tools/llamacpp-lora-converter
python convert_lora_to_gguf.py --outfile adapter.gguf --outtype q8_0 ../../output_adapters/qa_adapter
```

Then copy the adapter.gguf on top of the already existing adapter.gguf in the PAW cache, for the program that you compiled at the beginning.

```bash
cp  ~/.cache/programasweights/programs/<program_id>/adapter.gguf
```

Finally, drop the stale prefix KV cache

```bash
rm -f ~/.cache/programasweights/programs/<program_id>/prefix_kv_cache.bin
```

### TODO
1. polish `train_paw_lora_adapter.py`, needs logging too
2. make the procedure of converting LoRA and copying, dropping stuff, automated? In a Python pipeline.py?
3. once the whole process is complete, would it be worth to move the PAW cache inside this repo? And then what would one need to change in the PAW architecture to make it work in the future for QA?
4. implement the real QA
    a. `paw.function(<program_id>)` with the `program_id` for which you just trained the LoRA
    b. understand how to make it work giving PAW a database as input: it has to retrieve information from a database in the end

NOTE: continute from the chat: https://chat.mistral.ai/work/ebbce174-49bf-4a22-9ab5-5a66619927fc