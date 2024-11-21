class Foo:
    def __init__(self, x):
        self.x = x


if __name__ == '__main__':
    x = 1
    f1 = Foo(x)
    f2 = Foo(x)
    print(id(f1.x), id(f2.x))
    f1.x += 1  # New ID because integers are immutable
    print(id(f1.x), id(f2.x))
