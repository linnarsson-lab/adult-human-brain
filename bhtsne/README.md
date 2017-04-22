
# bhtsne

Modified from https://github.com/lvdmaaten/bhtsne as follows:

* Allow specifying the initial layout.
* Allow specifying an arbitrary (sparse) similarity matrix.

## Compiling

```bash
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```

## Usage

1. Make sure `bhtsne` is in your $PATH

2. Use the TSNE() wrapper class in `layout.py`

