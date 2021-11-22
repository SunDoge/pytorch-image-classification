import dataclasses
from typing import Type
import inspect

@dataclasses.dataclass
class B():

    foo: str
    bar: str
    foo1: str = '1'

    # def __init__(
    #     self,
    #     foo: str,
    #     bar: str,
    #     foo1,
    # ) -> None:
    #     pass


def create_from_locals(klass: Type):
    # keys = [sig for sig in inspect.signature(klass)]
    # print(keys)
    # kwargs = {k: local_vars[k] for k in keys if k in local_vars}
    # return klass(**kwargs)
    sig = inspect.signature(klass)
    keys = list(sig.parameters.keys())
    print(keys)
    # kwargs = {k: local_vars[k] for k in keys if k in local_vars}
    frame_info = inspect.stack()[1]
    local_vars = frame_info.frame.f_locals
    kwargs = {k: local_vars[k] for k in keys if k in local_vars}
    return klass(**kwargs)


def main():
    foo = '1'
    bar = '2'
    zoo = '3'

    foo1 = 'foo1'

    b = create_from_locals(B)
    print(b)


main()
