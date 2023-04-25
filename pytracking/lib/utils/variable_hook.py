import torch
from bytecode import Bytecode, Instr


class get_local(object):
    cache = {}
    is_activate = False

    def __init__(self, varname):
        self.varname = varname

    def __call__(self, func):
        if not type(self).is_activate:
            return func

        type(self).cache[func.__qualname__] = []
        c = Bytecode.from_code(func.__code__)
        extra_code = [
            Instr('STORE_FAST', '_res'),
            Instr('LOAD_FAST', self.varname),
            Instr('STORE_FAST', '_value'),
            Instr('LOAD_FAST', '_res'),
            Instr('LOAD_FAST', '_value'),
            Instr('BUILD_TUPLE', 2),
            Instr('STORE_FAST', '_result_tuple'),
            Instr('LOAD_FAST', '_result_tuple'),
        ]
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        def wrapper(*args, **kwargs):
            res, values = func(*args, **kwargs)
            if isinstance(values, torch.Tensor):
                type(self).cache[func.__qualname__].append(values.detach().cpu().numpy())
            elif isinstance(values, list):  # list of Tensor
                type(self).cache[func.__qualname__].append([value.detach().cpu().numpy() for value in values])
            else:
                raise NotImplementedError
            return res

        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True
