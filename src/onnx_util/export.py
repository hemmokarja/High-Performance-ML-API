import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import onnx
import structlog
import torch
from onnxconverter_common import float16

logger = structlog.get_logger(__name__)


def export_pytorch_to_onnx(
    model: torch.nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    output_path: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    use_fp16: bool = False,
    fp16_mode: str = "native",
    opset_version: int = 16,
) -> None:
    """
    Export any PyTorch model to ONNX format with optional FP16 precision.

    Args:
        model: PyTorch model in eval mode
        dummy_inputs: Dictionary of input tensors for tracing
        output_path: Where to save the .onnx file
        input_names: List of input names matching dummy_inputs keys
        output_names: List of output names
        dynamic_axes: Dict specifying which dimensions are dynamic
        use_fp16: Whether to convert model to FP16
        fp16_mode: "post_conversion" (convert after export) or "native" (export FP16
                   directly)
        opset_version: ONNX opset version
    """
    if fp16_mode not in ["native", "post_conversion"]:
        raise ValueError(
            f"Expected fp16_mode to be native or post_conversion, got {fp16_mode}"
        )

    logger.info("Starting ONNX export...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = str(output_path)

    model.eval()
    model = model.cpu()

    if use_fp16 and fp16_mode == "native":
        logger.info("Converting model to FP16 before export...")
        model = model.half()

    dummy_inputs_processed = {k: v.cpu() for k, v in dummy_inputs.items()}
    dummy_input_tuple = tuple(dummy_inputs_processed[name] for name in input_names)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input_tuple,
            output_path,
            export_params=True,  # store trained weights
            opset_version=opset_version,
            do_constant_folding=True,  # optimize constants
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            verbose=False,
        )

    logger.info(f"ONNX export complete, saved to: {output_path}")

    if use_fp16 and fp16_mode == "post_conversion":
        _convert_onnx_to_fp16(output_path)

    _validate_onnx_model(output_path)


def _convert_onnx_to_fp16(
    output_path: str,
    keep_io_types: bool = True,
    disable_shape_infer: bool = False,
    suppress_warnings: bool = True
) -> None:
    """
    Convert an ONNX model from FP32 to FP16.

    Args:
        input_path: Path to FP32 ONNX model
        output_path: Path to save FP16 ONNX model
        keep_io_types: Keep input/output in FP32 (recommended for compatibility)
        disable_shape_infer: Disable shape inference (use if conversion fails)
        suppress_warnings: Suppress value truncation warnings (default: True)
    """
    logger.info("Converting ONNX model to FP16...")
    model = onnx.load(output_path)

    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="onnxconverter_common"
            )
            model_fp16 = float16.convert_float_to_float16(
                model,
                keep_io_types=keep_io_types,
                disable_shape_infer=disable_shape_infer,
            )
    else:
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=keep_io_types,
            disable_shape_infer=disable_shape_infer,
        )

    onnx.save(model_fp16, output_path)
    logger.info(f"Successfully converted model to FP16, saved to: {output_path}")


def _log_shapes(elements):
    for elem in elements:
        shape = [
            d.dim_value if d.dim_value > 0 else d.dim_param
            for d in elem.type.tensor_type.shape.dim
        ]
        logger.info(f"  - {elem.name}: {shape}")


def _validate_onnx_model(onnx_path: str) -> None:
    """
    Validate the exported ONNX model structure.

    Args:
        onnx_path: Path to .onnx file
    """
    logger.info("Validating ONNX model...")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    logger.info("ONNX model is valid")

    logger.info("Model inputs:")
    _log_shapes(onnx_model.graph.input)

    logger.info("Model outputs:")
    _log_shapes(onnx_model.graph.output)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    logger.info(f"Model size: {size_mb:.2f} MB")
