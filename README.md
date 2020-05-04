# MGproc Weighted

A variation of mgproc implementing weighted metrics.

- The main difference from the original repository is in the metrics.py file.
- Weights need to be set manually in the metrics.py file (default: M1*1, M2*2).
- The tree_values.py file also includes the definition for **reactivation** metrics (metrics considering movement features).
- tree_values_sig.py can be alternatively chosen to use reactivation metrics modulated by a sigmoid function. In the current implementation, the file name needs to be manually switched to tree_values.py to be used, metrics having the same name. In a future refinement the two will just be merged.
- example trees include trees annotated with movement features that can be used by the reactivation metrics.
