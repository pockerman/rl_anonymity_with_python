
class Error(Exception):
    """
    General error class to handle generic errors
    """
    def __init__(self, message) -> None:
        self.message = message


class InvalidFileFormat(Exception):
    def __init__(self, filename):
        self.message = f"File {filename} has incorrect format."

    def __str__(self):
        return self.message


class InvalidParamValue(Exception):
    def __init__(self, param_name: str, param_value: str):
        self.message = "Parameter {0} has invalid value {1}".format(param_name, param_value)

    def __str__(self):
        return self.message


class InvalidSchemaException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self):
        return self.message






