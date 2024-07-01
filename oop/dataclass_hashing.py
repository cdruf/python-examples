from dataclasses import dataclass, field

my_set = set()


# Hashable and final
@dataclass(eq=True, frozen=True)
class Foo1:
    bar_1: int
    bar_2: int


my_set.add(Foo1(1, 2))


# Hashable and not final
@dataclass(eq=True, frozen=False, unsafe_hash=True)
class Foo2:
    bar_1: int = field(hash=True, compare=True)
    bar_2: int = field(hash=False, compare=False)


my_set.add(Foo2(1, 2))
my_set.add(Foo2(1, 3))  # not added because already in set

print(my_set)
