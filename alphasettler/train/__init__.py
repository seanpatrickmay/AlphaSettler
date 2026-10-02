"""Training the network (spec Section 4). Needs the `train` extra (PyTorch); nothing outside
this package imports torch. No numpy either: tensors come from the bindings' bytes through
`torch.frombuffer`, so torch's notice that numpy is missing is silenced here."""

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Failed to initialize NumPy", category=UserWarning)
    import torch  # noqa: F401
