import argparse
import os

import structlog
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

from onnx_util import export

load_dotenv()

logger = structlog.get_logger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")

DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1", "y")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX Exporter for HuggingFace models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name to export to ONNX"
    )
    parser.add_argument(
        "--use-fp16",
        type=_str_to_bool,
        default=False,
        help="Use FP16 instead of FP32"
    )
    return parser.parse_args()


def _make_output_path(model_name):
    model_id = model_name.replace("/", "_")
    return f"./onnx_models/huggingface/{model_id}.onnx"


def main():
    args = _parse_args()

    logger.info(f"Starting to export model {args.model_name} to ONNX...")

    model = AutoModel.from_pretrained(args.model_name, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dummy_text = ["This is a sample sentence for model tracing."]
    encoded = tokenizer(
        dummy_text, padding=True, truncation=True, return_tensors="pt"
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state", "pooler_output"]

    dummy_inputs = {name: encoded[name] for name in input_names}

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        "pooler_output": {0: "batch_size"}
    }

    output_path = _make_output_path(args.model_name)

    export.export_pytorch_to_onnx(
        model=model,
        dummy_inputs=dummy_inputs,
        output_path=output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        use_fp16=args.use_fp16,
        fp16_mode="native",  # more stable
        opset_version=17
    )


if __name__ == "__main__":
    main()
