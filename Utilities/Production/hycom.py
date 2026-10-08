"""Compatibility entry point for the shared HYCOM forecast downloader."""

import sys
from GeneralUtilities.Data.Download import production_hycom_download as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
