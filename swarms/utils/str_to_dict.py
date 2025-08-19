import orjson
from typing import Dict, Any, Union, Optional
from datetime import datetime, date
import uuid


def str_to_dict(s: str, retries: int = 3, options: Optional[int] = None) -> Dict:
    """
    Converts a JSON string to dictionary with enhanced orjson functionality.

    Args:
        s (str): The JSON string to be converted.
        retries (int): The number of times to retry parsing the string in case of a JSONDecodeError. Default is 3.
        options (Optional[int]): orjson options for parsing. Defaults to None.

    Returns:
        Dict: The parsed dictionary from the JSON string.

    Raises:
        orjson.JSONDecodeError: If the string cannot be parsed into a dictionary after the specified number of retries.
    """
    for attempt in range(retries):
        try:
            # Use orjson.loads for faster JSON parsing
            return orjson.loads(s)
        except orjson.JSONDecodeError as e:
            if attempt < retries - 1:
                continue  # Retry on failure
            else:
                raise e  # Raise the error if all retries fail


def dict_to_str(data: Dict[str, Any], options: Optional[int] = None) -> str:
    """
    Converts a dictionary to JSON string with enhanced orjson functionality.

    Args:
        data (Dict[str, Any]): The dictionary to convert.
        options (Optional[int]): orjson options for serialization. Defaults to None.

    Returns:
        str: The JSON string representation of the dictionary.
    """
    if options is None:
        options = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_NAIVE_UTC
    
    return orjson.dumps(data, option=options).decode('utf-8')


def safe_json_parse(s: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with fallback to default value.

    Args:
        s (str): The JSON string to parse.
        default (Any): Default value to return if parsing fails.

    Returns:
        Any: Parsed JSON object or default value.
    """
    try:
        return orjson.loads(s)
    except (orjson.JSONDecodeError, TypeError):
        return default


def validate_json(s: str) -> bool:
    """
    Validate if a string is valid JSON.

    Args:
        s (str): The string to validate.

    Returns:
        bool: True if valid JSON, False otherwise.
    """
    try:
        orjson.loads(s)
        return True
    except (orjson.JSONDecodeError, TypeError):
        return False


def json_with_datetime_support(data: Dict[str, Any]) -> str:
    """
    Serialize dictionary with datetime support using orjson.

    Args:
        data (Dict[str, Any]): Dictionary containing datetime objects.

    Returns:
        str: JSON string with datetime objects properly serialized.
    """
    options = (
        orjson.OPT_INDENT_2 | 
        orjson.OPT_SORT_KEYS | 
        orjson.OPT_NAIVE_UTC | 
        orjson.OPT_OMIT_MICROSECONDS
    )
    return orjson.dumps(data, option=options).decode('utf-8')


def json_with_uuid_support(data: Dict[str, Any]) -> str:
    """
    Serialize dictionary with UUID support using orjson.

    Args:
        data (Dict[str, Any]): Dictionary containing UUID objects.

    Returns:
        str: JSON string with UUID objects properly serialized.
    """
    options = (
        orjson.OPT_INDENT_2 | 
        orjson.OPT_SORT_KEYS | 
        orjson.OPT_NAIVE_UTC | 
        orjson.OPT_OMIT_MICROSECONDS
    )
    return orjson.dumps(data, option=options).decode('utf-8')


def compact_json(data: Dict[str, Any]) -> str:
    """
    Serialize dictionary to compact JSON string.

    Args:
        data (Dict[str, Any]): Dictionary to serialize.

    Returns:
        str: Compact JSON string without indentation.
    """
    return orjson.dumps(data, option=orjson.OPT_SORT_KEYS).decode('utf-8')


def pretty_json(data: Dict[str, Any], indent: int = 2) -> str:
    """
    Serialize dictionary to pretty-printed JSON string.

    Args:
        data (Dict[str, Any]): Dictionary to serialize.
        indent (int): Number of spaces for indentation.

    Returns:
        str: Pretty-printed JSON string.
    """
    if indent == 2:
        options = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
    elif indent == 4:
        options = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS  # orjson only supports 2-space indentation
    else:
        options = orjson.OPT_SORT_KEYS
    
    return orjson.dumps(data, option=options).decode('utf-8')


def json_from_file(file_path: str, encoding: str = 'utf-8') -> Dict[str, Any]:
    """
    Load JSON from file using orjson.

    Args:
        file_path (str): Path to the JSON file.
        encoding (str): File encoding.

    Returns:
        Dict[str, Any]: Parsed JSON data.

    Raises:
        FileNotFoundError: If file doesn't exist.
        orjson.JSONDecodeError: If file contains invalid JSON.
    """
    with open(file_path, 'rb') as f:
        return orjson.loads(f.read())


def json_to_file(data: Dict[str, Any], file_path: str, options: Optional[int] = None) -> None:
    """
    Save JSON to file using orjson.

    Args:
        data (Dict[str, Any]): Data to save.
        file_path (str): Path to save the JSON file.
        options (Optional[int]): orjson options for serialization.

    Raises:
        IOError: If file cannot be written.
    """
    if options is None:
        options = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
    
    with open(file_path, 'wb') as f:
        f.write(orjson.dumps(data, option=options))


def merge_json_strings(*json_strings: str) -> Dict[str, Any]:
    """
    Merge multiple JSON strings into a single dictionary.

    Args:
        *json_strings: Variable number of JSON strings to merge.

    Returns:
        Dict[str, Any]: Merged dictionary.

    Raises:
        orjson.JSONDecodeError: If any string contains invalid JSON.
    """
    result = {}
    for json_str in json_strings:
        parsed = orjson.loads(json_str)
        if isinstance(parsed, dict):
            result.update(parsed)
    return result


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain other content.

    Args:
        text (str): Text that may contain JSON.

    Returns:
        Optional[Dict[str, Any]]: Extracted JSON object or None if not found.
    """
    import re
    
    # Find JSON-like patterns in text
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
        r'\[[^\[\]]*(?:\{[^{}]*\}[^\[\]]*)*\]',  # Arrays with objects
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return orjson.loads(match)
            except orjson.JSONDecodeError:
                continue
    
    return None
