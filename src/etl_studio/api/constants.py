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
    "groupby": {
        "name": "GroupBy",
        "description": "Group by specific columns and aggregations",
        "parameters": ["group_columns", "aggregations"]
    }
}

# Available aggregation functions for groupby operations
AGGREGATION_FUNCTIONS = [
    {"id": "sum", "name": "Sum", "description": "Sum of values"},
    {"id": "mean", "name": "Mean", "description": "Average of values"},
    {"id": "count", "name": "Count", "description": "Count of values"},
    {"id": "min", "name": "Min", "description": "Minimum value"},
    {"id": "max", "name": "Max", "description": "Maximum value"},
    {"id": "first", "name": "First", "description": "First value"},
    {"id": "last", "name": "Last", "description": "Last value"},
    {"id": "std", "name": "Std Dev", "description": "Standard deviation"},
    {"id": "var", "name": "Variance", "description": "Variance"},
]
