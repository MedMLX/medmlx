"""Entry point for ``python -m medmlx``; behaves like the ``medmlx`` command."""

import sys

from medmlx.cli import main

sys.exit(main())
