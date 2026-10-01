

# Start-Process `
#   -FilePath "C:\Users\m273826\miniconda3\envs\env_vision\python.exe" `
#   -ArgumentList "-u -m RUN_Preprocessing.test_09_models_1_forward_real.main" `
#   -WorkingDirectory "D:\Marcelo\python_projects\Planta_Daninha" `
#   -RedirectStandardOutput "D:\Marcelo\python_projects\Planta_Daninha\run.log" `
#   -RedirectStandardError "D:\Marcelo\python_projects\Planta_Daninha\run_error.log"


# Get-Content .\run.log -Wait




import torch

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("Versão CUDA do PyTorch:", torch.version.cuda)
print("Número de GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}:")
    print("  Nome:", torch.cuda.get_device_name(i))
    print("  Capability:", torch.cuda.get_device_capability(i))

if torch.cuda.is_available():
    print("\nGPU atual:", torch.cuda.current_device())
    print("Nome da GPU atual:", torch.cuda.get_device_name(torch.cuda.current_device()))