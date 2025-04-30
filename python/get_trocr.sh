#!/bin/bash

pip install --upgrade --upgrade-strategy eager optimum[onnxruntime]

optimum-cli export onnx --model microsoft/trocr-base-handwritten --task vision2seq-lm trocr_onnx/     