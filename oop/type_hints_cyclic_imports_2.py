from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oop.type_hints_cyclic_imports_1 import Bar


def foo(bar: Bar):
    print(f"Foo & {bar.bar}")
