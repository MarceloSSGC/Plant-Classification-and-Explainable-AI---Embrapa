import torch
import sys

gpu = int(sys.argv[1])

device = torch.device(f"cuda:{gpu}")

print("=" * 70)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", gpu, torch.cuda.get_device_name(gpu))
print("=" * 70)

sizes = [
    (1, 5, 960, 1280),
    (2, 5, 960, 1280),
    (4, 5, 960, 1280),
    (8, 5, 960, 1280),
]

for shape in sizes:

    # CPU apenas
    torch.manual_seed(123)

    x = torch.randn(shape, dtype=torch.float32)

    # CPU -> GPU
    y = x.to(device)

    torch.cuda.synchronize(device)

    # GPU -> CPU
    z = y.cpu()

    diff = torch.abs(x - z)

    print("\nshape:", shape)
    print(f"size: {x.numel() * 4 / 1024**2:.1f} MB")

    print("GPU finite:", torch.isfinite(y).all().item())
    print("GPU NaN:", torch.isnan(y).sum().item())
    print("GPU Inf:", torch.isinf(y).sum().item())

    print("equal:", torch.equal(x, z))
    print("allclose:", torch.allclose(x, z))

    print("different values:", (x != z).sum().item())
    print("max diff:", diff.max().item())
    print("mean diff:", diff.mean().item())

print("\nDone.")