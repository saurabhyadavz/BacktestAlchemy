from datetime import datetime, time
from typing import Union

BACKEND_URL = "http://127.0.0.1:8000"
YY_MM_DD_FORMAT = "%Y-%m-%d"


def string_to_datetime(date_string: str) -> datetime:
    """
    Converts a string to a datetime object.
    Args:
        date_string (str): A string representing a date in the format 'YYYY-MM-DD'

    Returns:
        datetime: A datetime object representing the output date.
    """
    try:
        return datetime.strptime(date_string, YY_MM_DD_FORMAT)
    except TypeError as e:
        print(f"Error in func(string_to_datetime): {str(e)}")


def datetime_to_str(date_obj: datetime.date) -> str:
    """
    Converts a datetime object to a string.
    Args:
        date_obj (datetime.date): A datetime.date representing a datetime object

    Returns:
        str: A string representing the output date.
    """
    try:
        return date_obj.strftime(YY_MM_DD_FORMAT)
    except TypeError as e:
        print(f"Error in func(datetime_to_str): {str(e)}")

def str_to_time_obj(input_str: str) -> Union[datetime.time, None]:
    """
    Convert string of format "hh:mm" to time object
    Args:
        input_str(str): input time string of format "hh:mm"

    Returns:
        datetime.time: returns time object
    """
    try:
        print(input_str)
        hours, minutes = map(int, input_str.split(":"))
        time_obj = time(hour=hours, minute=minutes)
    except ValueError:
        print("Invalid input format. Please enter a string in the format 'hh:mm'")
        return None
    return time_obj
