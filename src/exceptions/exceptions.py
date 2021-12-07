
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


