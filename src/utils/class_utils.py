def protect(*protected):
    """Returns a metaclass that protects all attributes given as strings"""

    class Protect(type):
        has_base = False

        def __new__(meta, name, bases, attrs):
            if meta.has_base:
                for attribute in attrs:
                    print("attrs", attrs)
                    if attribute in protected:
                        raise AttributeError(
                            'Overriding of "%s" not allowed.' % attribute
                        )
            meta.has_base = True
            klass = super().__new__(meta, name, bases, attrs)
            return klass

    return Protect
