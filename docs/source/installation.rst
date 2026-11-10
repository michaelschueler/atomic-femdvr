Installation
============

The latest release is available from `PyPI <https://pypi.org/project/atomic_femdvr>`_:

.. code-block:: console

    $ python3 -m pip install atomic_femdvr

``atomic-femdvr`` requires Python 3.10 or newer. To plot wavefunctions with ``--plot``,
install the ``plot`` extra:

.. code-block:: console

    $ python3 -m pip install "atomic_femdvr[plot]"

Exchange-correlation functionals (libxc)
----------------------------------------

The built-in exchange-correlation driver (``"driver": "internal"``) provides the PBE
functional, which covers most pseudopotentials. Any other LDA or GGA functional, in
particular the LDA used for the NIST benchmarks, requires the Python bindings of
`libxc <https://libxc.gitlab.io>`_ (``"driver": "pylibxc"``). libxc is not on PyPI;
build it from source (requires CMake and a C compiler), see also the
`libxc installation guide <https://libxc.gitlab.io/installation/>`_:

.. code-block:: console

    $ python3 -m pip install "pylibxc @ git+https://gitlab.com/libxc/libxc.git"

Development version
-------------------

.. code-block:: console

    $ git clone https://github.com/michaelschueler/atomic-femdvr.git
    $ cd atomic-femdvr
    $ python3 -m pip install -e .
    $ python3 -m pytest tests

The NIST benchmark test is skipped unless ``pylibxc`` is installed.
