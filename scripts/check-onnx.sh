#!/bin/sh

USE_ONNX=${1:-false}

if [ "$USE_ONNX" = "true" ]; then
    if ! find onnx_models -type f -name "*.onnx" | grep -q .; then
        echo 'ERROR: USE_ONNX=true but no .onnx files found in ./onnx_models directory. Have you executed `make onnx-export`?'
        exit 1
    fi
fi
