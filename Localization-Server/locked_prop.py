class Test:
    def __init__(self):
        self._prop = ""

    def locked_property(func):
        def wrapped_prop(self, *args, **kwargs):
            print('before')
            value = func(self, *args, **kwargs)
            print('after')
            return value
        return wrapped_prop

    @property
    @locked_property
    def prop(self):
        print('get')
        return self._prop

    @prop.setter
    @locked_property
    def prop(self, value):
        print('set', value)
        self._prop = value


