from dataclasses import dataclass, field


@dataclass
class Foo:
    bar: int
    calculated_int: int = field(init=False)

    def __post_init__(self):
        self.calculated_int = self.bar * 2
        self.calculated_dict = {1: 2, 3: 4}


if __name__ == '__main__':
    foo = Foo(bar=1)
    print(foo)
    dct = foo.calculated_dict.copy()
    print(dct)
