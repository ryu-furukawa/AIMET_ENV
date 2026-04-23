import onnx
from onnx import helper, TensorProto, checker

model_path = "/root/AIMET_ruri/ruri-onnx/model.onnx"
out_path   = "/root/AIMET_ruri/ruri-onnx/model_dummy_input.onnx"

m = onnx.load(model_path)
g = m.graph

def ensure_input(name, elem_type=TensorProto.INT64, shape=["batch_size","sequence_length"]):
    if name not in [i.name for i in g.input]:
        g.input.append(helper.make_tensor_value_info(name, elem_type, shape))

ensure_input("token_type_ids")
ensure_input("position_ids")

if not any(init.name == "zero_i64" for init in g.initializer):
    g.initializer.append(helper.make_tensor("zero_i64", TensorProto.INT64, [], [0]))

target = "attention_mask"
keep_name = target + "_keep"

mul_tt  = helper.make_node("Mul", ["token_type_ids", "zero_i64"], ["tt_zero"],  name="TT_Zero")
mul_pos = helper.make_node("Mul", ["position_ids",   "zero_i64"], ["pos_zero"], name="POS_Zero")
add_0   = helper.make_node("Add", ["tt_zero", "pos_zero"], ["dummy_zero"], name="DummyZeroSum")
add_t   = helper.make_node("Add", [target, "dummy_zero"], [keep_name],     name="AddDummyToMask")

# 最初に attention_mask を使っているノードの直前に差し込む
first_consumer = None
for idx, node in enumerate(g.node):
    if target in node.input:
        first_consumer = idx
        break
if first_consumer is None:
    raise RuntimeError("attention_mask is not used anywhere")

# 追加ノードをその位置に挿入（順序を保つ）
new_nodes = [mul_tt, mul_pos, add_0, add_t]
for n in reversed(new_nodes):
    g.node.insert(first_consumer, n)

# ★重要：置換は「挿入したノード群より後ろだけ」実施する
start_replace = first_consumer + len(new_nodes)
for node in g.node[start_replace:]:
    for i, inp in enumerate(node.input):
        if inp == target:
            node.input[i] = keep_name

checker.check_model(m)
onnx.save(m, out_path)
print("Saved:", out_path)
