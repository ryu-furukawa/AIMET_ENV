import numpy as np
import torch
import torch.nn.functional as F


# ファイルパスを指定
#file1_path = '/root/AIMET_ENV/tools/comp_raw/ruri_0_pooled.raw'
#file1_path = '/root/AIMET_ENV/tools/comp_raw/ruri_0_seq.raw'
file1_path = '/root/AIMET_ENV/tools/comp_raw/mask_0_pooled_pos.raw'


#file1_path = '/root/jmteb/emu/output_2/Result_0/pooled_output.raw'
#file2_path = '/root/AIMET_ENV/tools/comp_raw/mask_0_pooled_pos_pad.raw'
file2_path = '/root/AIMET_ENV/tools/comp_raw/lock_fix_0_pool.raw'
#file2_path = '/root/AIMET_ENV/tools/comp_raw/pooled_output_normal.raw'  # ← 適宜変更
#file2_path = '/root/AIMET_ENV/tools/comp_raw/last_hidden_state_normal_nopad.raw'  # ← 適宜変更


# ファイルの内容を読み込む
data1 = np.fromfile(file1_path, dtype=np.float32)
data2 = np.fromfile(file2_path, dtype=np.float32)

with np.printoptions(threshold=100, edgeitems=3, precision=4, suppress=True):
    
    print(data1.shape)
    print(data1)
    print(data2.shape)
    print(data2)
    data2 = data2[:data1.shape[0]]  # data1 と同じ長さに切る
    print(data2.shape)
    print(data2)

x = torch.from_numpy(data1)
y = torch.from_numpy(data2)

sim = F.cosine_similarity(x, y, dim=0)
sim = torch.clamp(sim, -1.0, 1.0)

print(f'Cosine Similarity (PyTorch): {sim.item()}')

