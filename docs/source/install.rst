Installation
============

The following packages are required:

- `NumPy <https://numpy.org/>`_
- `Sphinx <https://www.sphinx-doc.org/en/master/>`_
- `Python Pandas <https://pandas.pydata.org/>`_
- `PyTorch <https://pytorch.org/>`_
- `Coverage.py <https://coverage.readthedocs.io/en/6.3.2/>`_

You can install there as usual with ``pip``.

.. code-block:: console

	pip install -r requirements.txt
	
Installation of the package is done via ``setuptools``

.. code-block:: console

   python setup.py
	
Run tests
---------

The is a series of tests to verify the implementation. You can executed these by running the script ``execute_tests_with_coverage.sh``.

Generate documentation
----------------------

You will need `Sphinx <https://www.sphinx-doc.org/en/master/>`_ in order to generate the API documentation. Assuming that Sphinx is already installed
on your machine execute the following commands (see also `Sphinx tutorial <https://www.sphinx-doc.org/en/master/tutorial/index.html>`_). 

.. code-block:: console

	sphinx-quickstart docs
	sphinx-build -b html docs/source/ docs/build/html



