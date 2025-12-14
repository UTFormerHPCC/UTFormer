# UTFormer

 UTFormer is a research codebase that implements a Transformer-based model for packet-level traffic classification. The repository contains model code, training and evaluation scripts, and preprocessing tools to convert pcap files into tensor datasets for training.

---

## Key Features

- Transformer-based model (MTT_dropping) with token dropping mechanisms to reduce computation overhead while maintaining classification performance.
- Preprocessing utilities to clean pcaps, extract per-packet hex payloads, and convert them into PyTorch tensors.
- Training and evaluation scripts written in PyTorch, with helpful debug utilities and metrics logging.

---

## Repository Structure

- `main/` — Core model and training scripts
  - `UTFormer.py` — Main entrypoint for building/training/evaluating the model
  - `learning_op.py` — Training and evaluation loops
  - `mtt_1.py` — Implementation of the MTT model and token-dropping transformer blocks
  - `mtt_block.py` — Additional transformer blocks used by the model
  - `traffic_loader.py` — PyTorch Dataset loader for preprocessed datasets
- `pre_process/` — Preprocessing tools for pcap files and dataset creation
  - `clean_pcap.py` — Filters and cleans pcaps using tshark
  - `pcap2bin.py` — Converts pcap packets into per-packet hex text lines
  - `make_label.py` — Generates label files for datasets
  - `pkt_level_tensor.py` — Utilities to convert hex data into PyTorch tensors
  - `README.md` — Details and usage for preprocessing pipeline
- `model/` — (Optional) model weights or training checkpoints (not tracked here)

---

## Requirements

- Python 3.8 or newer
- PyTorch (compatible version for your CUDA or CPU environment)
- scapy (for packet parsing in preprocessing)
- tqdm (progress bars)
- torcheval (metrics used in the training / evaluation scripts)
- tshark (for `clean_pcap.py` usage; install Wireshark on the host machine)

Install via pip (recommended inside a virtual environment):

Windows PowerShell:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch scapy tqdm torcheval
```

Linux/macOS:

```bash
python3 -m venv .venv; source .venv/bin/activate
pip install --upgrade pip
pip install torch scapy tqdm torcheval
```

---

## Preprocessing Pipeline

The `pre_process` folder provides a small pipeline to prepare datasets from pcap files. Typical steps:

1. Clean raw pcaps using `tshark` with `clean_pcap.py`.
2. Convert cleaned pcaps to per-packet hex text files using `pcap2bin.py`.
3. Generate label files using `make_label.py`.
4. Merge and convert hex files to PyTorch tensors using functions in `pkt_level_tensor.py`.

For detailed examples and commands, please see `pre_process/README.md`.

---

## Train & Evaluate

The training and evaluation entrypoint is `main/UTFormer.py`. The script builds an MTT_dropping model and then runs either training or evaluation depending on the parameters `pretrain` and `eval_only`.

Usage examples (edit the `main(...)` invocation inside `main/UTFormer.py` to set `pretrain` and `eval_only` flags):

Train model from scratch (default behavior if `pretrain=False` & `eval_only=False`):

```powershell
python main/UTFormer.py
```

To run only evaluation (make sure to point the model to a valid saved checkpoint — the script currently loads a file under `./best/` by name):

```powershell
python main/UTFormer.py
```

Note: The `UTFormer.py` script currently uses hard-coded dataset paths (for example, `'/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/train_data.pt'`). Update these paths to point to your dataset tensors before running training or evaluation.

If you prefer a CLI, you can create a thin wrapper around `UTFormer.py` or refactor the script to accept command-line arguments (argparse) to control flags and dataset paths.

---

## Model Architecture

The code implements a token-dropping transformer (MTT_dropping) where less-important tokens are pruned dynamically to reduce computation. The key modules:

- CustomEmbedding — converts byte tokens into fixed-size embeddings.
- PositionalEncoding — added to embeddings for positional information.
- TokenDropAttention / TokenDropBlock — attention modules with token dropping logic.
- MTT_Block — stacked transformer blocks for classification.

---

## Notes, Caveats & Tips

- Many scripts assume dataset tensor files in `.pt` format (PyTorch tensors). Use the tools in `pre_process/` to create these tensor files.
- Some path concatenations in scripts assume trailing path separators; provide directory paths ending with `/` or `\\` depending on your OS.
- Check GPU availability before training; if not present, the scripts will try to use CPU, but training might be considerably slower.
- Consider adjusting hyperparameters (batch size, number of epochs, learning rate, token keep rate) in `UTFormer.py` and `learning_op.py` for your dataset.

---


## License

Specify the project's license here (e.g., MIT, Apache 2.0), or add a license file to the repository.

---




## Accuracy and Performance

| Model        | Model type               | # Layer | # Head | Hidden dimension | Byte-keeping rate | Model size | Computation cost |
| ------------ | ------------------------ | ------- | ------ | ---------------- | ----------------- | ---------- | ---------------- |
|              |                          |         |        |                  |                   |            |                  |
| UTF-ISCX     | Light-weight Transformer | 1       | 4      | 268              | 5%                | 156 KB     | 241.8 KFLOPs     |
| UTF-USTC-app | Light-weight Transformer | 1       | 1      | 64               | 5%                | 106 KB     | 156.6 KFLOPs     |
| UTF-USTC-mal | Light-weight Transformer | 1       | 1      | 72               | 5%                | 107 KB     | 160.0 KFLOPs     |
| UTF-android  | Light-weight Transformer | 4       | 4      | 224              | 25%               | 8.8 MB     | 122.8 MFLOPs     |
| UTF-iOS      | Light-weight Transformer | 4       | 4      | 256              | 15%               | 12.8 MB    | 87.3 MFLOPs      |
| UTF-CapVideo | Light-weight Transformer | 1       | 1      | 64               | 10%               | 106 KB     | 180.4 KFLOPs     |

| Datasets    | USTC-TFC (app) |       |       |       | USTC-TFC (mal) |       |       |       | ISCX-VPN-2016 |       |       |       |
| ----------- | -------------- | ----- | ----- | ----- | -------------- | ----- | ----- | ----- | ------------- | ----- | ----- | ----- |
| Metrics (%) | RC             | PR    | F1    | AC    | RC             | PR    | F1    | AC    | RC            | PR    | F1    | AC    |
| UTFormer    | 99.78          | 99.66 | 99.74 | 99.69 | 99.99          | 99.99 | 99.99 | 99.99 | 94.02         | 91.01 | 93.24 | 94.35 |

| Datasets    | Cross-platform-android |       |       |       | Cross-platform-iOS |       |       |       | CmpVideo |       |       |         |
| ----------- | ---------------------- | ----- | ----- | ----- | ------------------ | ----- | ----- | ----- | -------- | ----- | ----- | ------- |
| Metrics (%) | RC                     | PR    | F1    | AC    | RC                 | PR    | F1    | AC    | RC       | PR    | F1    | AC      |
| UTFormer    | 86.70                  | 85.33 | 86.29 | 87.79 | 87.96              | 87.01 | 87.63 | 89.87 | 97.23    | 95.71 | 96.21 | 96.88  |
