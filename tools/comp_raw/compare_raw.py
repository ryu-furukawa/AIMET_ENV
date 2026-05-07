import numpy as np
import torch
import torch.nn.functional as F


# ファイルパスを指定
file1_path = '/root/AIMET_ENV/tools/comp_raw/ruri_0_pooled.raw'
#file2_path = '/root/AIMET_ENV/tools/comp_raw/onnx.raw'

#file1_path = '/root/jmteb/emu/output_2/Result_0/pooled_output.raw'
file2_path = '/root/AIMET_ENV/tools/comp_raw/onnx_0_pooled_pos.raw'



# ファイルの内容を読み込む
data1 = np.fromfile(file1_path, dtype=np.float32)
data2 = np.fromfile(file2_path, dtype=np.float32)

with np.printoptions(threshold=100, edgeitems=3, precision=4, suppress=True):
    
    print(data1.shape)
    print(data1)
    print(data2.shape)
    print(data2)

x = torch.from_numpy(data1)
y = torch.from_numpy(data2)

sim = F.cosine_similarity(x, y, dim=0)
sim = torch.clamp(sim, -1.0, 1.0)

print(f'Cosine Similarity (PyTorch): {sim.item()}')

