from dataclasses import dataclass


@dataclass
class Foo:
    f: int


@dataclass
class Bar(Foo):
    b: int


if __name__ == '__main__':
    b = Bar(1, 2)
    print(b)
