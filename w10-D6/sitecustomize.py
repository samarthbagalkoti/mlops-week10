# sitecustomize.py
# Auto-imported by Python if present on sys.path (CWD is on sys.path).
# Provides a safe alias for 'distutils' on Python 3.12+ using setuptools._distutils.
import sys
try:
    import distutils  # noqa: F401
except Exception:
    try:
        import setuptools._distutils as _distutils  # type: ignore
        sys.modules['distutils'] = _distutils
        try:
            from setuptools._distutils import spawn as _spawn  # type: ignore
            sys.modules['distutils.spawn'] = _spawn
        except Exception:
            pass
    except Exception:
        pass

