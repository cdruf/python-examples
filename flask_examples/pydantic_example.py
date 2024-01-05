import pydantic


class MyQueryModel(pydantic.BaseModel):
    some_num: int
    list_of_floats: pydantic.conlist(float, min_length=3, max_length=5)


@pydantic.validate_arguments
def my_function(data: MyQueryModel):
    print(data)


if __name__ == '__main__':
    data = MyQueryModel(some_num=1, list_of_floats=[1.1, 1.2, 1.3])
    my_function(data)
