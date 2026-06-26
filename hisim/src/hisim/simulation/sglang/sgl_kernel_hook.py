import os

# Bypass flashinfer version mismatch check for SGLang 0.5.13 compatibility
os.environ['FLASHINFER_DISABLE_VERSION_CHECK'] = '1'

from hisim.hook import BaseHook


class M_SGLangKernelLoadUtilHook(BaseHook):
    HOOK_CLASS_NAME = ""
    HOOK_MODULE_NAME = "sgl_kernel.load_utils"

    @classmethod
    def hook(cls, target):
        def override_load_architecture_specific_ops(*args, **kwargs):
            """
            ImportError:
            [sgl_kernel] CRITICAL: Could not load any common_ops library!
            """
            pass

        target._load_architecture_specific_ops = override_load_architecture_specific_ops
