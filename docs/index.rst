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
   :maxdepth: 2
   :caption: Get Started

   0_get_start/0_quick_start.md
   0_get_start/1_installations.md
   0_get_start/2_git.md
   0_get_start/3_pypi.md
   0_get_start/4_docker.md
   0_get_start/5_conda.md

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   1_tutorials/0_train.md
   1_tutorials/1_validation.md


.. toctree::
   :maxdepth: 2
   :caption: Model Zoo

   2_model_zoo/0_object_detection.md
   2_model_zoo/1_segmentation.md
   2_model_zoo/2_classification.md

.. toctree::
   :maxdepth: 2
   :caption: Custom YOLO

   3_custom/0_model.md
   3_custom/1_data_augment.md
   3_custom/2_loss.md
   3_custom/3_task.md


.. toctree::
   :maxdepth: 2
   :caption: Deploy

   4_deploy/1_deploy.md
   4_deploy/2_onnx.md
   4_deploy/3_tensorrt.md


.. toctree::
   :maxdepth: 2
   :caption: Deploy

   4_deploy/1_deploy.md
   4_deploy/2_onnx.md
   4_deploy/3_tensorrt.md

License
-------

YOLO is provided under the MIT License, which allows extensive freedom for reuse and distribution. See the LICENSE file for full license text.
