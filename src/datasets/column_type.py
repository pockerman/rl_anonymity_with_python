"""Module column_type specifies an enumeration of the
column. This is similar to the ARX software.
See the ARX documentation at:
https://arx.deidentifier.org/wp-content/uploads/javadoc/current/api/org/deidentifier/arx/AttributeType.html

"""

import enum


class ColumnType(enum.IntEnum):

    INVALID_TYPE = 0
    IDENTIFYING_ATTRIBUTE = 1
    SENSITIVE_ATTRIBUTE = 2
    INSENSITIVE_ATTRIBUTE = 3
    QUASI_IDENTIFYING_ATTRIBUTE = 4
