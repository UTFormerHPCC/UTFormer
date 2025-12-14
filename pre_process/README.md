# UTFormer — pre_process tools

This folder contains a set of small Python scripts for preprocessing network packet capture (pcap) files and converting them into text/binary formats suitable for downstream model training.

Supported language: Python

Main scripts

- `clean_pcap.py` — Filters and cleans pcap files using `tshark`. The script removes common noise packets (ARP, DNS, ICMP, TLS, etc.) and writes cleaned pcaps. It operates on files in a directory.
- `pcap2bin.py` — Uses `scapy` to read pcap files and extract each IP packet's raw bytes (or the first N bytes). Each packet is saved as a hex string (one packet per line) into a `.txt` file.
- `make_label.py` — Generates label files corresponding to data files. For each input data file it writes repeated label lines (one label per packet/line) and increments the label value per file.
- `pkt_level_tensor.py` — Collection of functions to merge text data, split datasets, convert hex-text to PyTorch tensors, and save/merge tensors. The file contains multiple helper routines for common workflows.
- `multi_class.py` — Small script stub that demonstrates loading a label tensor and inspecting its shape.

Typical workflow

1) Clean raw pcaps using `tshark`:

PowerShell example (ensure path ends with a trailing backslash `\\`):

```powershell
python clean_pcap.py C:\\path\\to\\raw_pcaps\\ C:\\path\\to\\clean_pcaps\\
```

Linux/macOS:

```bash
python3 clean_pcap.py /path/to/raw_pcaps/ /path/to/clean_pcaps/
```

2) Convert cleaned pcaps to per-packet hex text files:

```bash
python pcap2bin.py /path/to/clean_pcaps/ /path/to/bin_txts/
```

3) Generate label files for the datasets:

```bash
python make_label.py /path/to/bin_txts/ /path/to/labels/ 0
```

4) Merge and convert text data to PyTorch tensors (use routines in `pkt_level_tensor.py`):

```bash
python pkt_level_tensor.py /path/to/bin_txts/ /path/to/output_tensors/ 0 dataset_name
```

Dependencies

- Python 3.8+
- scapy — used by `pcap2bin.py` and `pkt_level_tensor.py` (`pip install scapy`)
- PyTorch — used by tensor conversion and scripting (`pip install torch`)
- tshark — required by `clean_pcap.py` (install Wireshark/tshark on the host)

Installation example (recommended: virtual environment)
PowerShell:

```powershell
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install --upgrade pip; pip install scapy torch
```

Linux/macOS:

```bash
python3 -m venv .venv; source .venv/bin/activate; pip install --upgrade pip; pip install scapy torch
```

Notes and caveats

- Many scripts concatenate paths with simple string addition (for example `src_path + file`). To avoid path problems, pass directory paths that end with a path separator (e.g. `/path/to/dir/` or `C:\\path\\to\\dir\\`).
- The pcap filtering expression used in `clean_pcap.py` is defined in the `clean_option` variable — adjust it to include/exclude protocols according to your dataset.
- In `pcap2bin.py`, packets are padded or truncated to `kept_length` (default 1500 bytes). Change `kept_length` if you need a different packet size.
- `pkt_level_tensor.py` includes many helper functions; review and call the specific routine you need for merging, splitting, or tensor conversion.

Debug suggestions

- Confirm `tshark` is available on the command line and that Python packages `scapy` and `torch` are installed in the active environment.
- Use the built-in `pdb` imports available in the scripts to step through processing when debugging.

Next steps I can help with

- Add a small CLI wrapper around `pkt_level_tensor.py` to expose the most common conversions with simple flags.
- Convert path concatenation in scripts to use `os.path.join` for robust path handling.
