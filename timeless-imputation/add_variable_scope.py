import functools
import tensorflow as tf

# Copyright 2017 Quim Llimona. https://pastebin.com/zM5c9xqX
def add_variable_scope(name=None, reuse=None):
    """Creates a variable_scope that contains all ops created by the function.
    The scope will default to the provided name or to the name of the function
    in CamelCase. If the function is a class constructor, it will default to
    the class name. It can also be specified with name='Name' at call time.
    """
    def _variable_scope_decorator(func):
        _name = name
        if _name is None:
            _name = func.__name__
            if _name == '__init__':
                _name = func.__class__.__name__
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            # Local, mutable copy of `name`.
            name_to_use = _name
            if 'name' in kwargs:
                if kwargs['name'] is not None:
                    name_to_use = kwargs['name']
                del kwargs['name']

            to_reuse = reuse
            if 'reuse' in kwargs:
                if kwargs['reuse'] is not None:
                    to_reuse = kwargs['reuse']
                del kwargs['reuse']

            with tf.variable_scope(name_to_use, reuse=to_reuse):
                return func(*args, **kwargs)
        return _wrapper
    return _variable_scope_decorator
