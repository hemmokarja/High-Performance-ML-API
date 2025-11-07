import os
from pathlib import Path
from typing import Dict, List, Optional

import onnx
import structlog
import torch

logger = structlog.get_logger(__name__)


def export_pytorch_to_onnx(
    model: torch.nn.Module,
    dummy_inputs: Dict[str, torch.Tensor],
    output_path: str,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = 16,
) -> None:
    """
    Export any PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode
        dummy_inputs: Dictionary of input tensors for tracing
                      e.g.,
                      {"input_ids": torch.zeros(...), "attention_mask": torch.ones(...)}
        output_path: Where to save the .onnx file
        input_names: List of input names matching dummy_inputs keys
        output_names: List of output names
        dynamic_axes: Dict specifying which dimensions are dynamic
                      e.g., {"input_ids": {0: "batch", 1: "sequence"}}
        opset_version: ONNX opset version
    """
    logger.info("Starting ONNX export...")
    output_path_ = Path(output_path)
    output_path_.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model = model.cpu()

    dummy_inputs_cpu = {k: v.cpu() for k, v in dummy_inputs.items()}
    dummy_input_tuple = tuple(dummy_inputs_cpu[name] for name in input_names)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input_tuple,
            output_path_,
            export_params=True,  # store trained weights
            opset_version=opset_version,
            do_constant_folding=True,  # optimize constants
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            verbose=False,
        )

    logger.info(f"ONNX export complete, saved to: {output_path}")
    _validate_onnx_model(output_path)


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
