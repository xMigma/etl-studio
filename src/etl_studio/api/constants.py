"""Constants and configurations for the API."""

SILVER_OPERATIONS = {
    "fillna": {
        "name": "FillNA",
        "description": "Fill null values",
        "requires_value": True
    },
    "drop_nulls": {
        "name": "DropNulls",
        "description": "Remove rows with null values",
        "requires_value": False
    },
    "drop_duplicates": {
        "name": "DropDuplicates",
        "description": "Remove duplicate rows",
        "requires_value": False
    },
    "lowercase": {
        "name": "Lowercase",
        "description": "Convert text to lowercase",
        "requires_value": False
    },
    "rename_column": {
        "name": "RenameColumn",
        "description": "Rename column",
        "requires_value": True
    },
    "drop_column": {
        "name": "DropColumn",
        "description": "Delete column",
        "requires_value": False
    },
    "cast_type": {
        "name": "CastType",
        "description": "Change data type",
        "requires_value": True
    }
}
