from .version import __version__

from . import defaults
from .defaults import (
    DEFAULT_COMPRESSION,
    DEFAULT_CNAME,
    DEFAULT_CLEVEL,
    DEFAULT_SHUFFLE,
)

from .schema import ColumnSchema, TableSchema

from . import util

from . import metafile

# Import modules that depend on the C extension only if it's available
try:
    from . import _column_pywrap  # type: ignore[attr-defined]
    from . import _column

    from . import column
    from .column import Column

    from . import columns
    from .columns import Columns

    from . import chunks

    from . import indices
    from .indices import Indices

    from . import mergesort

    from . import convenience

    _C_EXTENSION_AVAILABLE = True
except ImportError:
    _C_EXTENSION_AVAILABLE = False
