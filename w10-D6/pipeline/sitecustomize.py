# sitecustomize.py
# Shim for Python 3.12 where 'distutils' was removed but some libs still import it.
import sys
try:
    import distutils  # noqa: F401
except Exception:
    try:
        import setuptools._distutils as _distutils  # type: ignore
        sys.modules['distutils'] = _distutils
        # expose submodules commonly used, e.g. distutils.spawn
        try:
            from setuptools._distutils import spawn as _spawn  # type: ignore
            sys.modules['distutils.spawn'] = _spawn
        except Exception:
            pass
    except Exception:
        # If setuptools isn't present or changed internals, we simply don't shim.
        # But with our requirements pin, this path shouldn't hit.
        pass

