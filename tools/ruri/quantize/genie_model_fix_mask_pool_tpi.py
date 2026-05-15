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
    ensure_input(graph, "token_type_ids")

    if not any(init.name == "zero_i64" for init in graph.initializer):
        graph.initializer.append(
            helper.make_tensor("zero_i64", TensorProto.INT64, [], [0])
        )

    target = "attention_mask"
    keep_name = target + "_keep"

    mul_tt = helper.make_node(
        "Mul", ["token_type_ids", "zero_i64"], ["tt_zero"], name="TT_Zero"
    )
    add_t = helper.make_node(
        "Add", [target, "tt_zero"], [keep_name], name="AddDummyToMask"
    )

    first_consumer = None
    for idx, node in enumerate(graph.node):
        if target in node.input:
            first_consumer = idx
            break
    if first_consumer is None:
        raise RuntimeError("attention_mask is not used anywhere")

    new_nodes = [mul_tt, add_t]
    for n in reversed(new_nodes):
        graph.node.insert(first_consumer, n)

    start_replace = first_consumer + len(new_nodes)
    for node in graph.node[start_replace:]:
        for i, inp in enumerate(node.input):
            if inp == target:
                node.input[i] = keep_name

    return keep_name

def add_masked_mean_pooling(graph, sequence_output: str, mask_name: str, pooled_output: str):

    # ---- axes initializer（opset13用） ----
    axes_unsq = "axes_unsqueeze"
    axes_red = "axes_reduce"

    if not any(init.name == axes_unsq for init in graph.initializer):
        graph.initializer.append(
            helper.make_tensor(axes_unsq, TensorProto.INT64, [1], [2])
        )

    if not any(init.name == axes_red for init in graph.initializer):
        graph.initializer.append(
            helper.make_tensor(axes_red, TensorProto.INT64, [1], [1])
        )

    # ---- eps ----
    eps_name = "eps_f"
    if not any(init.name == eps_name for init in graph.initializer):
        graph.initializer.append(
            helper.make_tensor(eps_name, TensorProto.FLOAT, [1], [1e-6])
        )

    # ---- mask cast ----
    mask_f = "mask_f"
    cast_node = helper.make_node(
        "Cast",
        inputs=[mask_name],
        outputs=[mask_f],
        to=TensorProto.FLOAT
    )

    # ---- unsqueeze ----
    mask_u = "mask_u"
    unsq_node = helper.make_node(
        "Unsqueeze",
        inputs=[mask_f, axes_unsq],
        outputs=[mask_u]
    )

    # ---- mask適用 ----
    masked_seq = "masked_seq"
    mul_node = helper.make_node(
        "Mul",
        inputs=[sequence_output, mask_u],
        outputs=[masked_seq]
    )

    # ---- sum seq ----
    sum_seq = "sum_seq"
    sum_seq_node = helper.make_node(
        "ReduceSum",
        inputs=[masked_seq, axes_red],   # ★修正済み
        outputs=[sum_seq],
        keepdims=0
    )

    # ---- sum mask ----
    sum_mask = "sum_mask"
    sum_mask_node = helper.make_node(
        "ReduceSum",
        inputs=[mask_u, axes_red],       # ★修正済み
        outputs=[sum_mask],
        keepdims=0
    )

    # ---- denom ----
    denom = "denom"
    denom_node = helper.make_node(
        "Add",
        inputs=[sum_mask, eps_name],
        outputs=[denom]
    )

    # ---- div ----
    div_node = helper.make_node(
        "Div",
        inputs=[sum_seq, denom],
        outputs=[pooled_output]
    )

    graph.node.extend([
        cast_node, unsq_node, mul_node,
        sum_seq_node, sum_mask_node,
        denom_node, div_node
    ])


def main(in_path: str, out_path: str):
    model = onnx.load(in_path)
    graph = model.graph

    # --- 1) token_type_ids / position_ids の dummy input 追加 + attention_mask_keep 作成 ---
    mask_keep_name = add_dummy_inputs_and_keep_attention_mask(graph)  # "attention_mask_keep"

    # --- 2) last_hidden_state を探す ---
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

    # --- 3) last_hidden_state -> sequence_output ---
    sequence_output = "sequence_output"
    identity_node = helper.make_node(
        "Identity",
        inputs=[target_output_name],
        outputs=[sequence_output],
        name="Identity_last_hidden_state_to_sequence_output",
    )
    graph.node.extend([identity_node])

    # --- 4) pooling_output を追加（mask付きmean pooling） ---
    pooled_output = "pooled_output"
    add_masked_mean_pooling(graph, sequence_output, mask_keep_name, pooled_output)

    # --- 5) 公開出力の差し替え ---
    remove_graph_output(graph, "last_hidden_state")
    remove_graph_output(graph, sequence_output)
    remove_graph_output(graph, pooled_output)

    seq_vi = helper.make_tensor_value_info(
        sequence_output, TensorProto.FLOAT, ['batch_size', 'sequence_length', 768]
    )
    # ★ batchは固定しない方がshape mergeのwarningを避けやすい
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