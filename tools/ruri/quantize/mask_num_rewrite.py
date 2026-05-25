import onnx
import numpy as np
from onnx import numpy_helper

INPUT_MODEL = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_mask_add.onnx"
OUTPUT_MODEL = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_mask_add_-10.onnx"

OLD = np.float32(-3.4028235e38)
NEW = np.float32(-10.0)

model = onnx.load(INPUT_MODEL)
patched_count = 0

# Constant ノードを書き換え
for node in model.graph.node:
    if node.op_type != "Constant":
        continue

    for attr in node.attribute:
        if attr.name != "value":
            continue

        arr = numpy_helper.to_array(attr.t)

        # scalar float32 のみ対象
        if arr.shape == () and arr.dtype == np.float32:
            if arr.item() == OLD:
                new_arr = np.array(NEW, dtype=np.float32)
                new_tensor = numpy_helper.from_array(new_arr, name=attr.t.name)
                attr.t.CopyFrom(new_tensor)
                patched_count += 1
                print(f"patched Constant node: {node.name or '(no name)'}")

# initializer を書き換え
for i, init in enumerate(model.graph.initializer):
    arr = numpy_helper.to_array(init)

    if arr.shape == () and arr.dtype == np.float32:
        if arr.item() == OLD:
            new_arr = np.array(NEW, dtype=np.float32)
            model.graph.initializer[i].CopyFrom(
                numpy_helper.from_array(new_arr, name=init.name)
            )
            patched_count += 1
            print(f"patched initializer: {init.name}")

onnx.checker.check_model(model)
onnx.save(model, OUTPUT_MODEL)

print(f"done. patched_count={patched_count}")
print(f"saved to: {OUTPUT_MODEL}")