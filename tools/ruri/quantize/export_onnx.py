# export_onnx.py
import os
import inspect
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "/root/AIMET_ruri/ruri-small-v2"
OUT_DIR = "/root/AIMET_ruri_v2/ruri-onnx_vvv"
SEQ_LEN = 512
OPSET = 17


# -------------------------------------------------
# SDPA mask を安全側に差し替え（存在する場合のみ）
# -------------------------------------------------
try:
    import transformers.masking_utils as mu

    def sdpa_mask_safe(attention_mask, *args, **kwargs):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            # [B,S] -> [B,1,1,S] bool
            return attention_mask[:, None, None, :].to(torch.bool)
        return attention_mask.to(torch.bool)

    if hasattr(mu, "sdpa_mask"):
        mu.sdpa_mask = sdpa_mask_safe
except Exception:
    pass


class ExportWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

        if hasattr(self.base, "config"):
            self.base.config.use_cache = False

        # forward が受け取れる引数を確認
        sig = inspect.signature(self.base.forward)
        self.accepted_args = set(sig.parameters.keys())

    def forward(self, input_ids, attention_mask):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": False,
        }

        out = self.base(**kwargs)
        return out[0]  # last_hidden_state


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------
    # tokenizer（重要）
    # -----------------------
    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # -----------------------
    # model（eager attention）
    # -----------------------
    try:
        base = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except TypeError:
        base = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

    base.eval()

    if hasattr(base, "set_attn_implementation"):
        try:
            base.set_attn_implementation("eager")
        except Exception:
            pass

    model = ExportWrapper(base).eval()

    # -----------------------
    # dummy input
    # -----------------------
    dummy = tok(
        ["hello world"],
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN,
        return_tensors="pt",
    )

    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    #token_type_ids = dummy.get(
    #    "token_type_ids",
    #    torch.zeros_like(input_ids),
    #)


    
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "last_hidden_state": {0: "batch_size", 1: "seq_len"},
    }

    # -----------------------
    # ONNX export（legacy）
    # -----------------------
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        f"{OUT_DIR}/model.onnx",
        opset_version=OPSET,
        input_names=[
            "input_ids",
            "attention_mask",
            #"token_type_ids",  # ruri は通常 token_type_ids を受け取らない想定
        ],
        output_names=[
            "last_hidden_state",
        ],
        dynamic_axes=dynamic_axes,
        #dynamic_axes=None,
        do_constant_folding=True,
        dynamo=False,  # ★ legacy exporter
    )

    tok.save_pretrained(OUT_DIR)
    print("✅ Exported:", f"{OUT_DIR}/model.onnx")


if __name__ == "__main__":
    with torch.inference_mode():
        main()