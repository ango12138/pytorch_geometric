import os
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

import torch
from packaging import version


def is_full_test() -> bool:
    r"""Whether to run the full but time-consuming test suite."""
    return os.getenv('FULL_TEST', '0') == '1'


def onlyFullTest(func: Callable) -> Callable:
    r"""A decorator to specify that this function belongs to the full test
    suite."""
    import pytest
    return pytest.mark.skipif(
        not is_full_test(),
        reason="Fast test run",
    )(func)


def withPython(*args) -> Callable:
    r"""A decorator to skip tests for any Python version not listed."""
    def decorator(func: Callable) -> Callable:
        import pytest

        python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
        return pytest.mark.skipif(
            python_version not in args,
            reason=f"Python {python_version} not supported",
        )(func)

    return decorator


def matches_version(package: str, base_version: str):
    if base_version is None:
        return True

    package_version = import_module(package).__version__
    return version.parse(package_version) >= version.parse(base_version)


def withPackage(*args) -> Callable:
    r"""A decorator to skip tests if certain packages are not installed. Also
    supports version information, only with >=."""
    versions_split = [arg.split('>=') for arg in args]
    na_packages = set(
        arg[0] for arg in versions_split if find_spec(arg[0]) is None
        or not matches_version(arg[0], arg[1] if len(arg) > 1 else None))

    def decorator(func: Callable) -> Callable:
        import pytest
        return pytest.mark.skipif(
            not is_full_test() and len(na_packages) > 0,
            reason=f"Package(s) {na_packages} are not installed",
        )(func)

    return decorator


def withRegisteredOp(*args) -> Callable:
    r"""A decorator to skip tests if a certain op is not registered."""
    def is_registered(op: str) -> bool:
        module = torch.ops
        for attr in op.split('.'):
            try:
                module = getattr(module, attr)
            except RuntimeError:
                return False
        return True

    na_ops = set(arg for arg in args if not is_registered(arg))

    def decorator(func: Callable) -> Callable:
        import pytest

        return pytest.mark.skipif(
            len(na_ops) > 0,
            reason=f"Operator(s) {na_ops} are not registered",
        )(func)

    return decorator


def withCUDA(func: Callable) -> Callable:
    r"""A decorator to skip tests if CUDA is not found."""
    import pytest
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )(func)
