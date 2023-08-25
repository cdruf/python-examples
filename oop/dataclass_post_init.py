from dataclasses import dataclass, field


@dataclass
class Foo:
    bar: int
    calculated_field: int = field(init=False)

    def __post_init__(self):
        self.calculated_field = self.bar * 2


if __name__ == '__main__':
    foo = Foo(bar=1)
    print(foo)
