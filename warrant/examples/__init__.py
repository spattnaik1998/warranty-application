"""Demo graphs shipped inside the package.

These live under ``warrant/`` rather than the repo's top-level ``examples/`` so
that ``warrant audit --example ...`` works after a plain ``pip install warrant``
— the top-level directory is not packaged (and ``examples`` is far too generic a
name to claim on PyPI). The runnable scripts that *drive* these graphs stay in
the repo's ``examples/``, where they are documentation rather than product.
"""
