# export_onnx.py
# Run once to produce navantix_densenet.onnx
# Requires: pip install onnx onnxruntime

import os
import torch
import torchxrayvision as xrv
import numpy as np

print("Loading DenseNet-121...")
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# Fixed batch=1. No dynamic_axes — torchxrayvision's pooling layer
# doesn't export cleanly with symbolic batch dims.
dummy = torch.randn(1, 1, 224, 224)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "navantix_densenet.onnx")
print(f"Exporting to {out_path}...")

torch.onnx.export(
    model,
    dummy,
    out_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)

print("Exported. Verifying with onnxruntime...")
import onnxruntime as ort
session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
onnx_out = session.run(None, {"input": dummy.numpy()})[0]
print(f"ONNX output shape: {onnx_out.shape}")

with torch.no_grad():
    torch_out = model(dummy).numpy()

max_diff = float(np.abs(onnx_out - torch_out).max())
print(f"Max diff vs PyTorch: {max_diff:.6f}")

if max_diff < 1e-3:
    print("✅ Export verified. Ready to use.")
else:
    print("⚠️  Diff is larger than expected. Investigate before using.")