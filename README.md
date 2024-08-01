# tf-sha256

`tf-sha256` is a Python library for computing SHA256 with TensorFlow.
[SHA256](https://en.wikipedia.org/wiki/SHA-2) is a common cryptographic hash function. This library provides an implementation of SHA256 with TensorFlow, which means it enables us to potentially design loss functions with it to train ML models.

## Installation

Use the package manager [poetry](https://python-poetry.org/) to install `tf-sha256`.

```bash
poetry install
```

Note I haven't published this package to Pypi since in my own experience, the compatibility of TensorFlow and CUDA varies, so it might be easier for you to install this directly from the repo, or even copy-paste/submodule the files to your project, such that you can tweak things for your setup.

## Usage

Please refer to the [unit tests](tests/test_sha256.py) for a more detailed usage of the library.
Note this library has not been used in any production workload, despite reaching parity with non-tf implementation of SHA256 (as shown by the unit tests).

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

Consider [sponsoring](https://github.com/sponsors/Jason-Y-Z) the project if this turns out useful☕

## License

[MIT](https://choosealicense.com/licenses/mit/)
