class InvalidArgumentError(Exception):
    """Error to be used when the arguments of a method or class are invalid.

    Parameters
    ----------
    argument : any
        Argument that is invalid.
    message : str
        Message to be shown as an error.
    """
    def __init__(self, argument, message):
        self.argument = argument
        self.message = message
