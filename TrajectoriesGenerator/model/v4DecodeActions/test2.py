class Test:
    def __init__(self) -> None:
        pass

    def pintar(self) -> None:
        print(a)


def my_func(*args):
    for arg in args:
        print(arg)

def my_func(**kwargs):
    for name, arg in kwargs.items():
        print(f"{name}: {arg}")

my_func(int1=3,int2=2,mi_string="asdas",lista=[1,2,3,4])