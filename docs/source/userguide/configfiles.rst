

********************
Configuration Files
********************

Three types of input files are needed for the configuration. The ``picasso_config_file``, the ``polly_config_file`` (both with their respective template files),
and a table (``pollynet_config_link_file``) linking the systems, times and ``polly_config_files``. 

They are loaded with the following logic, producing three outputs: ``picasso_config_dict``, ``polly_config_array``, ``polly_config_dict``:


.. image:: img/PicassoPy_config_overview.png
    :width: 60%

With calling :py:meth:`~ppcpy.interface.picassoProc.PicassoProc`, the two config dicts are combined with the raw data to the ``data_cube``.
They are stored as variables in the ``data_cube``:

.. code-block:: python

    data_cube.polly_config_dict
    data_cube.picasso_config_dict


picasso_config_file
===================

+-----------------+-------------+
| Key             | Description |
+=================+=============+
| ``path_config`` | ....        |
+-----------------+-------------+
| ...             | ...         |
+-----------------+-------------+

polly_config_file
=================



