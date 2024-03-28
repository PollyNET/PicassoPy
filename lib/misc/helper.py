from pathlib import Path

def detect_path_type(fullpath):
    """
    Detect the type of path (Windows or Linux) based on the input.
    """
    def is_windows_path(fullpath):
        if '\\' in str(fullpath):
            return True
        else:
            return False

    def is_linux_path(fullpath):
        if '/' in str(fullpath):
            return True
        else:
            return False
        
    if is_windows_path(fullpath):
        # Parse the drive letter and path
        drive_letter, path_part = str(fullpath).split(':', 1)
        # Remove leading backslash from path_part
        path_part = path_part.lstrip('\\')
        path_object = Path(drive_letter + ':' + '\\' + path_part)
        return path_object
    elif is_linux_path(fullpath):
        path_object = Path(fullpath)
        return path_object
    else:
        return "Unknown OS-type"

