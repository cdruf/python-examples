from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from oop.type_hints_cyclic_imports_1 import Bar


def foo(bar: Bar):
    print(f"Foo & {bar.bar}")


@dataclass
class X:
    bar: Dict[int, Bar]
