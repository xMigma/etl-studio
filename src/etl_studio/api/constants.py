"""Constants and configurations for the API."""

SILVER_OPERATIONS = {
    "fillna": {
        "name": "FillNA",
        "description": "Fill null values",
        "parameters": ["column", "value"]
    },
    "drop_nulls": {
        "name": "DropNulls",
        "description": "Remove rows with null values",
        "parameters": ["column"]
    },
    "drop_duplicates": {
        "name": "DropDuplicates",
        "description": "Remove duplicate rows",
        "parameters": ["column"]
    },
    "lowercase": {
        "name": "Lowercase",
        "description": "Convert text to lowercase",
        "parameters": ["column"]
    },
    "rename_column": {
        "name": "RenameColumn",
        "description": "Rename column",
        "parameters": ["column", "new_name"]
    },
    "drop_column": {
        "name": "DropColumn",
        "description": "Delete column",
        "parameters": ["column"]
    },
    "group_by_column": {
        "name": "GroupByColumn",
        "description": "Group by a specific column",
        "parameters": ["column"]
    }
}
