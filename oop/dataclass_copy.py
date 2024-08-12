from dataclasses import fields, dataclass


@dataclass
class Foo:
    a: int
    b: float

    def copy(self):
        ret = Foo(a=0, b=0.0)
        for field in fields(Foo):
            setattr(ret, field.name, getattr(self, field.name))
        return ret


foo = Foo(a=10, b=10.0)
bar = foo.copy()
bar.a = 20
bar.b = 20.0
print(foo)
print(bar)
