#!/usr/bin/env python3
"""Stop the OpenAI client from forking a subprocess on every request.

`openai==1.10.0` stamps an `x-stainless-arch` header on each request. It builds
that header with `get_architecture()`, which carries no cache and calls
`platform.architecture()`; on Python 3.9 that shells out to `file` through
`subprocess`. So every chat completion forks a child process.

In a small process the fork costs a millisecond and nobody notices. In these
runners the process has habitat, torch and a sentence encoder resident -- six
hundred threads and tens of gigabytes of address space -- and the fork is issued
from a pool worker. Three times now one of those forks has wedged while holding
the GIL, and the whole run stops: `num_requests_running` on the endpoint drops to
zero, the output file stays at zero bytes, and the torch threads keep burning CPU
outside the GIL, so from the outside it is indistinguishable from slow work. The
three hangs cost 3h, 10h and 10h of wall clock; the last one was caught with
py-spy, stuck in `_execute_child` under `platform.architecture`.

The architecture of a running interpreter does not change, so the answer is read
once, before any worker thread exists, and returned from then on. The header value
is byte-identical to what the unpatched client would have sent, so cells produced
with this installed stay comparable with the ones produced before it.
"""

from __future__ import annotations

import platform
from typing import Any, Tuple

_frozen: Tuple[str, str] | None = None


def install() -> Tuple[str, str]:
    """Read the architecture once, single-threaded, and freeze it.

    Call this before the first client is built and before any pool is started.
    Idempotent, so a runner that imports another runner cannot double-install.
    """
    global _frozen
    if _frozen is None:
        _frozen = platform.architecture()  # the one fork this process pays for

        def architecture(*_args: Any, **_kwargs: Any) -> Tuple[str, str]:
            return _frozen  # type: ignore[return-value]

        platform.architecture = architecture
    return _frozen
