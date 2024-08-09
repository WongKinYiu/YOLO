Install YOLO
============

This guide will help you set up YOLO on your machine.
We recommend starting with `GitHub Settings <#git-github>`_ for more flexible customization.
If you are planning to perform inference only or require a simple customization, you can choose to install via `PyPI <#pypi-pip-install>`_.

Torch Requirements
-------------------

The following table summarizes the torch requirements for different operating systems and hardware configurations:


.. tabs::

   .. tab:: Linux

      .. tabs::

         .. tab:: CUDA

            PyTorch: 1.12+

         .. tab:: CPU

            PyTorch: 1.12+

   .. tab:: MacOS

      .. tabs::

         .. tab:: MPS

            PyTorch: 2.2+
         .. tab:: CPU
            PyTorch: 2.2+
   .. tab:: Windows

      .. tabs::

         .. tab:: CUDA

            [WIP]

         .. tab:: CPU

            [WIP]


Git & GitHub
------------

First, Clone the repository:

.. code-block:: bash

   git clone https://github.com/WongKinYiu/YOLO.git

Alternatively, you can directly download the repository via this `link <https://github.com/WongKinYiu/YOLO/archive/refs/heads/main.zip>`_.

Next, install the required packages:

.. code-block:: bash

    # For the minimal requirements, use:
    pip install -r requirements.txt
    # For a full installation, use:
    pip install -r requirements-dev.txt

Moreover, if you plan to utilize ONNX or TensorRT, please follow :ref:`ONNX`, :ref:`TensorRT` for more installation details.

PyPI (pip install)
------------------

.. note::
    Due to the :guilabel:`yolo` this name already being occupied in the PyPI library, we are still determining the package name.
    Currently, we provide an alternative way to install via the GitHub repository. Ensure your shell has `git` and `pip3` (or `pip`).

To install YOLO via GitHub:

.. code-block:: bash

   pip install git+https://github.com/WongKinYiu/YOLO.git

Docker
------

To run YOLO using NVIDIA Docker, you can pull the Docker image and run it with GPU support:

.. code-block:: bash

   docker pull henrytsui000/yolo
   docker run --gpus all -it henrytsui000/yolo

Make sure you have the NVIDIA Docker toolkit installed. For more details on setting up NVIDIA Docker, refer to the `NVIDIA Docker documentation <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_.


Conda
-----

We will publish it in the near future!
