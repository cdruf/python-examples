import copy


class Foo:
    def __init__(self):
        self.x = 1


class Bar:
    def __init__(self):
        self.foo = Foo()


if __name__ == '__main__':
    bar = Bar()
    c = copy.deepcopy(bar)
    bar.foo.x = 2
    print(c.foo.x)
