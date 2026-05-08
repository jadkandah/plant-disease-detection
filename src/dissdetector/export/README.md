# Mobile Model Export

React Native cannot run raw PyTorch `.pth` checkpoints. Use this exporter to
convert one image-only checkpoint to ONNX, then bundle the ONNX model and the
generated manifest into `mobile-v2/assets/mobile_models/`.

Recommended offline candidate:

```bash
python -m src.dissdetector.export.export_mobile_model \
  --checkpoint saved_models/mobilenet_v3_small_512_epochs25_full_data_set.pth \
  --model-name mobilenet_v3_small \
  --image-size 512 \
  --dataset-root jordan_dataset \
  --dataset-variant original
```

Outputs:

- `mobile_models/offline_model.onnx`
- `mobile_models/offline_model_manifest.json`

The exporter:

- Loads the image-only PyTorch architecture from the existing model factory.
- Loads the `.pth` state dict.
- Prefers a saved manifest beside the model when one exists.
- Rebuilds `class_mapping` from the dataset when no manifest exists.
- Exports ONNX with input `image` and output `logits`.
- Validates with `onnxruntime` when installed.
- Refuses multimodal model names because backend metadata models are not valid
  offline mobile models.

After export, copy the outputs:

```bash
mkdir -p mobile-v2/assets/mobile_models
cp mobile_models/offline_model.onnx mobile-v2/assets/mobile_models/
cp mobile_models/offline_model_manifest.json mobile-v2/assets/mobile_models/
```

The React Native app is scaffolded around `onnxruntime-react-native`. Native
ONNX runtime setup requires a development build/prebuild; Expo Go is not enough.
