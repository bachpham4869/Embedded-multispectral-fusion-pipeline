"""Package entry shim for ``python -m smartbinocular``.

Delegates to :func:`smartbinocular.main.main`, which owns the full interactive
fusion runtime (camera threads, display loop, and teardown). No pipeline logic
lives in this module.
"""
from smartbinocular.main import main

if __name__ == "__main__":
    main()
