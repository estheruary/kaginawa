class KaginawaError(Exception):
    """The base exception for the kaginawa library.

    All exceptions will be derived from this class which means you can catch
    all (known) errors produced by this library with the following

        from kaginawa.exceptions import KaginawaError

        try:
            ...
        except KaginawaError:
            ...

    This library (which uses requests under the hood) swallows RequestException
    and reraises it with `raise KaginawaError(...) from RequestException(...)`
    so you don't have to worry about handling those errors directly but they are
    available if needed.
    """

    pass
