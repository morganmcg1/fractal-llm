"""
Inference-only shim for the `rustbpe` extension module.

NanoChat imports `rustbpe` unconditionally in `nanochat/tokenizer.py`, but the web
chat server only needs `tokenizer.pkl` + `token_bytes.pt` for inference.

`src/chat.py` prepends this directory to PYTHONPATH only when `rustbpe` is not
installed, so local chat can run without a Rust toolchain.
"""


class Tokenizer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "rustbpe is not installed (shim in use). This is fine for inference/chat, "
            "but tokenizer training requires building the real rustbpe extension."
        )

