import torch


def main():

    print("Your system has the following GPU(s) available:")

    for i in range(torch.cuda.device_count()):
        gpu_stats = torch.cuda.get_device_properties(i)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"- GPU {i+1} = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"    - {start_gpu_memory} GB of memory reserved.")


if __name__ == "__main__":
    main()