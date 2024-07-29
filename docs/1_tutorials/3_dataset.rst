Create Dataset
==============

In this section, we will prepare the dataset and create a dataloader.

Overall, the dataloader can be created by:

.. code-block:: python

   from yolo import create_dataloader
   dataloader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task, use_ddp)

For inference, the dataset will be handled by :class:`~yolo.tools.data_loader.StreamDataLoader`, while for training and validation, it will be handled by :class:`~yolo.tools.data_loader.YoloDataLoader`.

The input arguments are:

- **DataConfig**: :class:`~yolo.config.config.DataConfig`, the relevant configuration for the dataloader.
- **DatasetConfig**: :class:`~yolo.config.config.DatasetConfig`, the relevant configuration for the dataset.
- **task_name**: :guilabel:`str`, the task name, which can be `inference`, `validation`, or `train`.
- **use_ddp**: :guilabel:`bool`, whether to use DDP (Distributed Data Parallel). Default is `False`.

Train and Validation
----------------------------

Dataloader Return Type
~~~~~~~~~~~~~~~~~~~~~

For each iteration, the return type includes:

- **batch_size**: the size of each batch, used to calculate batch average loss.
- **images**: the input images.
- **targets**: the ground truth of the images according to the task.

Auto Download Dataset
~~~~~~~~~~~~~~~~~~~~~

The dataset will be auto-downloaded if the user provides the `auto_download` configuration. For example, if the configuration is as follows:


.. literalinclude:: ../../yolo/config/dataset/mock.yaml
  :language: YAML


First, it will download and unzip the dataset from `{prefix}/{postfix}`, and verify that the dataset has `{file_num}` files.

Once the dataset is verified, it will generate `{train, validation}.cache` in Tensor format, which accelerates the dataset preparation speed.

Inference
-----------------

In streaming mode, the model will infer the most recent frame and draw the bounding boxes by default, given the save flag to save the image. In other modes, it will save the predictions to `runs/inference/{exp_name}/outputs/` by default.

Dataloader Return Type
~~~~~~~~~~~~~~~~~~~~~

For each iteration, the return type of `StreamDataLoader` includes:

- **images**: tensor, the size of each batch, used to calculate batch average loss.
- **rev_tensor**: tensor, reverse tensor for reverting the bounding boxes and images to the input shape.
- **origin_frame**: tensor, the original input image.

Input Type
~~~~~~~~~~

- **Stream Input**:

  - **webcam**: :guilabel:`int`, ID of the webcam, for example, 0, 1.
  - **rtmp**: :guilabel:`str`, RTMP address.

- **Single Source**:

  - **image**: :guilabel:`Path`, path to image files (`jpeg`, `jpg`, `png`, `tiff`).
  - **video**: :guilabel:`Path`, path to video files (`mp4`).

- **Folder**:

  - **folder of images**: :guilabel:`Path`, the relative or absolute path to the folder containing images.
