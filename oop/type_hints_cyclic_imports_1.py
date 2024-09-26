from oop.type_hints_cyclic_imports_2 import foo


class Bar:
    def __init__(self):
        self.bar = 1


if __name__ == '__main__':
    b = Bar()
    foo(b)