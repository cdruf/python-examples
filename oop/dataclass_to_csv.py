from dataclasses import dataclass, fields


def get_csv_header(dc):
    return ', '.join(f.name for f in fields(dc))


def to_csv(dc):
    return ', '.join(str(getattr(dc, f.name)) for f in fields(dc))


@dataclass
class Foo:
    a: int
    b: float
    c: bool

    def get_csv_header(self):
        return ', '.join(f.name for f in fields(self))

    def to_csv(self):
        return ', '.join(str(getattr(self, f.name)) for f in fields(self))


if __name__ == "__main__":
    f1 = Foo(1, 2.3, True)
    f2 = Foo(4, 5.6, False)
    print(f1.get_csv_header())
    print(f1.to_csv())
    print(f2.to_csv())

    print("")
    print(get_csv_header(f1))
    print(to_csv(f1))
    print(to_csv(f2))
