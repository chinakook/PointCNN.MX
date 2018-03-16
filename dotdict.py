# coding: utf-8

class DotDict(dict):
    """
    Simple class for dot access elements in dict, support nested initialization
    Example:
    d = DotDict({'child': 'dotdict'}, name='dotdict', index=1, contents=['a', 'b'])
    # add new key
    d.new_key = '!' # or d['new_key'] = '!'
    # update values
    d.new_key = '!!!'
    # delete keys
    del d.new_key
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]