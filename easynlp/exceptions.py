class InvalidArgumentError(Exception):
    def __init__(self, argument, message):
        self.argument = argument
        self.message = message
