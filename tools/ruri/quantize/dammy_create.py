# genie_model.py
import onnx
from onnx import helper, TensorProto

def upsert_initializer(graph, tensor):
    idx = next((i for i, t in enumerate(graph.initializer) if t.name == tensor.name), None)
    if idx is None:
        graph.initializer.extend([tensor])
    else:
        graph.initializer[idx] = tensor

def remove_graph_output(graph, name: str):
    """graph.output から name を持つ ValueInfo を安全に削除"""
    idxs = [i for i, o in enumerate(graph.output) if o.name == name]
    for i in reversed(idxs):
        del graph.output[i]

def ensure_input(graph, name, elem_type=TensorProto.INT64, shape=["batch_size", "sequence_length"]):
    if name not in [i.name for i in graph.input]:
        graph.input.append(helper.make_tensor_value_info(name, elem_type, shape))

def add_dummy_inputs_and_keep_attention_mask(graph):
    # position_idsは削除 → token_type_idsのみ
    ensure_input(graph, "token_type_ids")

    if not any(init.name == "zero_i64" for init in graph.initializer):
        graph.initializer.append(helper.make_tensor("zero_i64", TensorProto.INT64, [], [0]))

    target = "attention_mask"
    keep_name = target + "_keep"

    # position系を削除
    mul_tt = helper.make_node(
        "Mul", ["token_type_ids", "zero_i64"], ["tt_zero"], name="TT_Zero"
    )

    # pos_zero削除 → DummyZeroSumも simplification
    add_0 = helper.make_node(
        "Add", ["tt_zero", "zero_i64"], ["dummy_zero"], name="DummyZeroSum"
    )

    add_t = helper.make_node(
        "Add", [target, "dummy_zero"], [keep_name], name="AddDummyToMask"
    )

    # --- 挿入位置 ---
    first_consumer = None
    for idx, node in enumerate(graph.node):
        if target in node.input:
            first_consumer = idx
            break
    if first_consumer is None:
        raise RuntimeError("attention_mask is not used anywhere")

    new_nodes = [mul_tt, add_0, add_t]  # POS削除
    for n in reversed(new_nodes):
        graph.node.insert(first_consumer, n)

    start_replace = first_consumer + len(new_nodes)
    for node in graph.node[start_replace:]:
        for i, inp in enumerate(node.input):
            if inp == target:
                node.input[i] = keep_name


def main(in_path: str, out_path: str):
    model = onnx.load(in_path)
    graph = model.graph

    # --- 1) token_type_ids / position_ids の dummy input 追加 ---
    add_dummy_inputs_and_keep_attention_mask(graph)

    # --- 1) last_hidden_state を探す ---
    target_output_name = None
    for o in graph.output:
        if o.name == "last_hidden_state":
            target_output_name = o.name
            break
    if target_output_name is None:
        for node in graph.node:
            if "last_hidden_state" in node.output:
                target_output_name = "last_hidden_state"
                break
    if target_output_name is None:
        raise RuntimeError("last_hidden_state が見つかりません。")

    # --- 2) 固定長化は廃止。Identity で sequence_output に流す ---
    sequence_output = "sequence_output"
    identity_node = helper.make_node(
        "Identity",
        inputs=[target_output_name],
        outputs=[sequence_output],
        name="Identity_last_hidden_state_to_sequence_output",
    )
    graph.node.extend([identity_node])


    # === 3) 平均プーリング（軸=1）を ReduceMean + Constant 入力で実装 =======
    # === 3) 平均プーリング（軸=1）を opset11 仕様で実装 =======
    pooled_output = "pooled_output"

    reduce_mean_node = helper.make_node(
        "ReduceMean",
        inputs=[sequence_output],   # ★ 入力は1つだけ
        outputs=[pooled_output],
        name="ReduceMean_tokens_axis1",
        axes=[1],                   # ★ opset11: 属性として指定
        keepdims=0
    )
    graph.node.extend([reduce_mean_node])

    #======================================================================


    # --- 5) 公開出力の差し替え ---
    remove_graph_output(graph, "last_hidden_state")
    remove_graph_output(graph, sequence_output)
    remove_graph_output(graph, pooled_output)

    # sequence_output は “型のみ（形状は未固定）” で公開
    seq_vi = helper.make_tensor_value_info(
        sequence_output, TensorProto.FLOAT, ['batch_size', 'sequence_length', 768]
    )
    pooled_vi = helper.make_tensor_value_info(
        pooled_output, TensorProto.FLOAT, [1, 768]
    )
    graph.output.extend([seq_vi, pooled_vi])

    # --- 6) 検証 & 保存 ---
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    args = p.parse_args()
    main(args.in_path, args.out_path)
