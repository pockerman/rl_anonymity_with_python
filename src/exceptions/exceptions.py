from typing import Any


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


class InvalidDataTypeException(Exception):
    def __init__(self, param_name: str, param_type: Any, param_types: str):
        self.message = "Parameter {0} has invalid type. Type {1} not in {2}".format(param_name, str(Any), param_types)

    def __str__(self):
        return self.message


class InvalidSchemaException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self):
        return self.message


class InvalidStateException(Exception):
    def __init__(self, type_name: str, state_type: str) -> None:
        self.message = "Type= {0} is not in state= {1}".format(type_name, state_type)

    def __str__(self):
        return self.message


class IncompatibleVectorSizesException(Exception):
    def __init__(self, size1: int, size2: int) -> None:
        self.message = "Size {0} does not match size {1} ".format(size1, size2)

    def __str__(self):
        return self.message





