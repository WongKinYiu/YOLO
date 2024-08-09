YOLO documentation
=======================

Introduction
------------

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system that is designed for both efficiency and accuracy. This documentation provides comprehensive guidance on how to set up, configure, and effectively use YOLO for object detection tasks.

**Note: This project and some sections of this documentation are currently a work in progress.**

Project Features
----------------

- **Real-time Processing**: YOLO can process images in real-time with high accuracy, making it suitable for applications that require instant detection.
- **Multitasking Capabilities**: Our enhanced version of YOLO supports multitasking, allowing it to handle multiple object detection tasks simultaneously.
- **Open Source**: YOLO is open source, released under the MIT License, encouraging a broad community of developers to contribute and build upon the existing framework.

Documentation Contents
----------------------

Explore our documentation:


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   0_get_start/0_quick_start
   0_get_start/1_introduction
   0_get_start/2_installations

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   1_tutorials/0_allIn1
   1_tutorials/1_setup
   1_tutorials/2_buildmodel
   1_tutorials/3_dataset
   1_tutorials/4_train
   1_tutorials/5_inference


.. toctree::
   :maxdepth: 1
   :caption: Model Zoo

   2_model_zoo/0_object_detection
   2_model_zoo/1_segmentation
   2_model_zoo/2_classification

.. toctree::
   :maxdepth: 1
   :caption: Custom YOLO

   3_custom/0_model
   3_custom/1_data_augment
   3_custom/2_loss
   3_custom/3_task


.. toctree::
   :maxdepth: 1
   :caption: Deploy

   4_deploy/1_deploy
   4_deploy/2_onnx
   4_deploy/3_tensorrt


.. toctree::
   :maxdepth: 1
   :caption: Features

   5_features/0_small_object
   5_features/1_version_convert
   5_features/2_IPython

.. toctree::
   :maxdepth: 1
   :caption: Function Docs

   6_function_docs/0_solver
   6_function_docs/1_tools
   6_function_docs/2_module

License
-------

YOLO is provided under the MIT License, which allows extensive freedom for reuse and distribution. See the LICENSE file for full license text.
