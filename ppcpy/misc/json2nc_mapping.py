import logging
import json
from netCDF4 import Dataset

def read_json_to_dict(file_path):
    """
    Reads in an existing json-file and outputs a dict-structure
    """
    with open(file_path, 'r') as file:
        data = json.load(file)  # Parse JSON into a dictionary
    return data


def create_netcdf_from_dict(nc_file_path, data_dict, compression_level=1):
    """
    Creates a NetCDF file from a structured dictionary.

    Args:
        nc_file_path (str): Path to the NetCDF file to create.
        data_dict (dict): Dictionary with keys 'global_attributes', 'dimensions', and 'variables'.

    Example of `data_dict` structure:
    {
        "global_attributes": {
            "title": "Example NetCDF File",
            "institution": "My Organization"
        },
        "dimensions": {
            "time": None,  # Unlimited dimension
            "lat": 10,
            "lon": 20
        },
        "variables": {
            "temperature": {
                "dimensions": ("time", "lat", "lon"),
                "dtype": "float32",
                "attributes": {
                    "units": "K",
                    "long_name": "Surface temperature"
                },
                "data": np.random.rand(5, 10, 20)  # Example data
            },
            "pressure": {
                "dimensions": ("time", "lat", "lon"),
                "dtype": "float32",
                "attributes": {
                    "units": "Pa",
                    "long_name": "Surface pressure"
                },
                "data": np.random.rand(5, 10, 20)  # Example data
            }
        }
    }
    """
    logging.info(f"writing to file: {nc_file_path}")
    # Create a new NetCDF file
    with Dataset(nc_file_path, 'w', format='NETCDF4') as nc_file:
        # Add global attributes
        if 'global_attributes' in data_dict:
            for attr_name, attr_value in data_dict['global_attributes'].items():
                setattr(nc_file, attr_name, attr_value)

        # Define dimensions
        if 'dimensions' in data_dict:
            for dim_name, dim_size in data_dict['dimensions'].items():
                nc_file.createDimension(dim_name, dim_size)

        # Define variables and add data
        if 'variables' in data_dict:
            for var_name, var_info in data_dict['variables'].items():
                # Extract variable metadata
                dimensions = var_info['shape']
                dtype = var_info['dtype']
                attributes = var_info.get('attributes', {})
                data = var_info.get('data')

                # Create variable
                var = nc_file.createVariable(var_name, dtype, dimensions,zlib=True,complevel=compression_level)

                # Add variable attributes
                for attr_name, attr_value in attributes.items():
                    setattr(var, attr_name, attr_value)

                # Add variable data (if provided)
                if data is not None:
                    var[:] = data


def add_variable_2_json_dict_mapper(data_dict, new_key, reference_key, new_data = None, new_attributes=None):
    """
    Adds a new variable to the 'variables' section of the given dictionary.
    
    Parameters:
        data_dict (dict): The original dictionary structure.
        reference_key (str): The name of the existing variable to reference to (template for new key/variable)
        new_key (str): The name of the new variable to add.
        new_data (np.ndarray, optional): The data for the new variable.
        new_attributes (dict, optional): Additional or updated attributes for the new variable.
    """
    # Ensure the new data dimensions match the existing structure
    if reference_key not in data_dict["variables"]:
        raise KeyError(f"Reference key '{reference_key}' not found in 'variables'.")

    # Copy the structure from the reference key
    new_variable = data_dict["variables"][reference_key].copy()
    
    # Update the data
    if new_data:
        new_variable["data"] = new_data
    
    # Update the attributes if provided
    if new_attributes:
        new_variable["attributes"].update(new_attributes)
    
    # Add the new variable to the dictionary
    data_dict["variables"][new_key] = new_variable


def remove_variable_from_json_dict_mapper(data_dict, key_to_remove):
    """
    Removes a specific variable from the 'variables' section of the given dictionary.
    
    Parameters:
        data_dict (dict): The original dictionary structure.
        key_to_remove (str): The name of the variable to remove.
    """
    if key_to_remove in data_dict["variables"]:
        del data_dict["variables"][key_to_remove]
    else:
        raise KeyError(f"Variable '{key_to_remove}' does not exist in 'variables'.")


def update_variable_attribute_of_json_dict_mapper(data_dict, variable_key, attribute_key, new_value):
    """
    Updates the value of a specific attribute for a specified variable in the data dictionary.

    Parameters:
        data_dict (dict): The dictionary containing the variables and attributes.
        variable_key (str): The key of the variable to update.
        attribute_key (str): The key of the attribute to update.
        new_value (any): The new value to set for the attribute.
    """
    # Check if the variable exists in the dictionary
    if variable_key not in data_dict["variables"]:
        raise KeyError(f"Variable '{variable_key}' not found in the dictionary.")
    
    # Check if the attribute exists in the variable's attributes
    if "attributes" not in data_dict["variables"][variable_key]:
        raise KeyError(f"'attributes' section not found in variable '{variable_key}'.")
    
    if attribute_key not in data_dict["variables"][variable_key]["attributes"]:
        raise KeyError(f"Attribute '{attribute_key}' not found in variable '{variable_key}' attributes.")
    
    # Update the attribute value
    data_dict["variables"][variable_key]["attributes"][attribute_key] = new_value


