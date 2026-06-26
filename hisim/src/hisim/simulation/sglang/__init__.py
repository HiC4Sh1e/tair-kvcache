import os

# Bypass flashinfer version mismatch check for SGLang 0.5.13 compatibility
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'
