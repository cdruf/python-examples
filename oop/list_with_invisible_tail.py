class MyList:

    def __init__(self):
        """The current index points to the last data point that is available at the current time. """
        self.lst = list(range(100))
        self.current_idx = 10

    def __getitem__(self, idx):
        length = self.current_idx + 1
        if isinstance(idx, int):
            if idx >= 0:
                if not (0 <= idx < length):
                    raise IndexError("Index out of range")
                return self.lst[idx]
            if idx < 0:
                i = length + idx
                if not (0 <= i < length):
                    raise IndexError("Index out of range")
                return self.lst[i]
        elif isinstance(idx, slice):
            if idx.start < 0 and idx.stop is None:  # get the tail
                i = length + idx.start
                if not (0 <= i < length):
                    raise IndexError("Index out of range")
                return self.lst[i:self.current_idx + 1]
            raise NotImplementedError(f"Slice not implemented: {idx.start}, {idx.stop}, {idx.step}")

        else:
            raise TypeError("Index must be an int or slice")


if __name__ == '__main__':
    lst = list(range(11))
    my_list = MyList()
    assert my_list[0] == lst[0]
    assert my_list[10] == lst[10]
    try:
        print(my_list[11])
    except IndexError:
        print("Index out of range")

    assert my_list[-1] == lst[-1]
    assert my_list[-11] == lst[-11]
    try:
        print(my_list[-12])
    except IndexError:
        print("Index out of range")

    assert my_list[-3:] == lst[-3:]
