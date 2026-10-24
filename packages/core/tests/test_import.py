def test_package_import_does_not_load_optional_frameworks() -> None:
    import subprocess
    import sys

    code = """
import importlib.abc
import sys

class BlockedFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'torch', 'monai'}:
            raise ImportError(f'blocked: {fullname}')

sys.meta_path.insert(0, BlockedFrameworks())
import medmlx_core
assert not {'mlx', 'torch', 'monai'} & sys.modules.keys()
assert medmlx_core.MedmlxError.__bases__ == (Exception,)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
