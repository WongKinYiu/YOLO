Quick Start
===========

.. note::
   We expect all customizations to be done primarily by passing arguments or modifying the YAML config files.
   If more detailed modifications are needed, custom content should be modularized as much as possible to avoid extensive code modifications.

.. _QuickInstallYOLO:

Install YOLO
------------

Clone the repository and install the dependencies:

.. code-block:: bash

   git clone https://github.com/WongKinYiu/YOLO.git
   cd YOLO
   pip install -r requirements-dev.txt
   # Make sure to work inside the cloned folder.

Alternatively, If you are planning to make a simple change:

**Note**: In the following examples, you should replace ``python yolo/lazy.py`` with ``yolo`` .

.. code-block:: bash

   pip install git+https://github.com/WongKinYiu/YOLO.git

**Note**: Most tasks already include at yolo/lazy.py, so you can run with this prefix and follow arguments: ``python yolo/lazy.py``


Train Model
-----------

To train the model, use the following command:

.. code-block:: bash

   python yolo/lazy.py task=train

   yolo task=train # if installed via pip

- Overriding the ``dataset`` parameter, you can customize your dataset via a dataset config.
- Overriding YOLO model by setting the ``model`` parameter to ``{v9-c, v9-m, ...}``.
- More details can be found at :ref:`Train Tutorials<Train>`.

For example:

.. code-block:: bash

   python yolo/lazy.py task=train dataset=AYamlFilePath model=v9-m

   yolo task=train dataset=AYamlFilePath model=v9-m # if installed via pip

Inference & Deployment
------------------------

Inference is the default task of ``yolo/lazy.py``. To run inference and deploy the model, use:
More details can be found at :ref:`Inference Tutorials <Inference>`.

.. code-block:: bash

   python yolo/lazy.py task.data.source=AnySource

   yolo task.data.source=AnySource # if installed via pip

You can enable fast inference modes by adding the parameter ``task.fast_inference={onnx, trt, deploy}``.

- Theoretical acceleration following :ref:`YOLOv9 <Deploy>`.
- Hardware acceleration like :ref:`ONNX <ONNX>` and :ref:`TensorRT <TensorRT>`. for optimized deployment.
