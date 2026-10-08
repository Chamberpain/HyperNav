"""Compatibility imports for the shared WCOFS ROMS preprocessor."""
from GeneralUtilities.Data.Download import wcofs_roms as _shared

globals().update({name: value for name, value in vars(_shared).items()
                  if not name.startswith('__')})
