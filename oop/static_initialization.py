from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class Foo:
    next_id: ClassVar[int] = field(default=0)
    id: int

    def __init__(self):
        self.id = Foo.next_id
        Foo.next_id += 1


if __name__ == '__main__':
    f1 = Foo()
    f2 = Foo()
    print(f1)
    print(f2)
