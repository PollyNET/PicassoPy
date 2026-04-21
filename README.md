## PollynetProcessingChain python version

[Documentation](https://pollynet.github.io/PicassoPy-doc/html/index.html)

### Installation

```
# setup a virtual environment (either conda or venv)
git clone https://github.com/PollyNET/PicassoPy.git

cd PicassoPy
python -m pip install -e .
```


### Usage

```
python3 tests\picassopy_testing.py --picasso_config_file "pollynet_processing_chain_config.json" \
    --date 20240308 --device pollyxt_cpv \
    --level0_file_to_process "2024_03_08_Fri_CPV_00_00_01.nc"
```

### Contribute
Your contribution is welcomed! Just have a look into the issue, some of them are quite easy to fix or file one yourself detailing a bug you found.
Additions to the documentation or tests is also always encouraged. Also, if you want to contribute more sophisticated algorithms feel free to file a pull request.

### History

### License
GNU GPL (see LICENSE)