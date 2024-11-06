import dataclasses
from dataclasses import dataclass, asdict, astuple


@dataclass
class Foo:
    x: int
    y: float


if __name__ == '__main__':
    f = Foo(1, 2.3)
    fields = {field.name: field.type for field in dataclasses.fields(f)}
    print(fields)

    field_names = [a for a in Foo.__dict__['__dataclass_fields__']]
    print(field_names)

    print(asdict(f))
    print(astuple(f))

    # Modify
    current_val = getattr(f, field_names[0])
    setattr(f, field_names[0], current_val + 100)
    print(f)
