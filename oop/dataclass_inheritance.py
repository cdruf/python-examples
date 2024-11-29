from dataclasses import dataclass, field
from typing import List, Callable


@dataclass
class Foo:
    f: int
    g: int
    callbacks: List[Callable] = field(init=False, repr=False)


@dataclass
class Bar(Foo):
    b: int

    def __post_init__(self):
        self.callbacks = [self.my_callback]

    def my_callback(self):
        print("Hi", self.b)


if __name__ == '__main__':
    b = Bar(1, 2, 3)
    print(b)
    for c in b.callbacks:
        c()
