## Instructions

From root of the project:

Compile PAW program and save it.

```bash
python -m run.paw_qa.compile_paw_program
```
Train LoRA adapter for future QA. 

IMPORTANT: you will be prompted to choose for which PAW program you want to train the adapter. It must be for the same that you have just compiled.

```bash
python -m run.paq_qa.train_paw_lora_adapter
```

Convert LoRA adapter weights to GGUF.

```bash
cd tools/llamacpp-lora-converter
python convert_lora_to_gguf.py --outfile adapter.gguf --outtype q8_0 ../../output_adapters/qa_adapter
```

Then copy the adapter.gguf on top of the already existing adapter.gguf in the PAW cache, for the program that you compiled at the beginning.

```bash
cp  ./output_adapter/adapter.gguf ~/.cache/programasweights/programs/<program_id>/adapter.gguf
```

Finally, drop the stale prefix KV cache

```bash
rm -f ~/.cache/programasweights/programs/<program_id>/prefix_kv_cache.bin
```

### TODO
NOTE: continute from the chat: https://chat.mistral.ai/work/ebbce174-49bf-4a22-9ab5-5a66619927fc