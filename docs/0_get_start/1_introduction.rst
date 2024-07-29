What is YOLO
============

``YOLO`` (You Only Look Once) is a state-of-the-art, real-time object detection system. It is designed to predict bounding boxes and class probabilities for objects in an image with high accuracy and speed. YOLO models, including the latest YOLOv9, are known for their efficiency in detecting objects in a single forward pass through the network, making them highly suitable for real-time applications.

YOLOv9 introduces improvements in both architecture and loss functions to enhance prediction accuracy and inference speed.

Forward Process
---------------

The forward process of YOLOv9 can be visualized as follows:

.. mermaid::

    graph LR
    subgraph YOLOv9
        Auxiliary
        AP["Auxiliary Prediction"]
    end
    BackBone-->FPN;
    FPN-->PAN;
    PAN-->MP["Main Prediction"];
    BackBone-->Auxiliary;
    Auxiliary-->AP;

- **BackBone**: Extracts features from the input image.
- **FPN (Feature Pyramid Network)**: Aggregates features at different scales.
- **PAN (Region Proposal Network)**: Proposes regions of interest.
- **Main Prediction**: The primary detection output.
- **Auxiliary Prediction**: Additional predictions to assist the main prediction.

Loss Function
-------------

The loss function of YOLOv9 combines several components to optimize the model's performance:

.. mermaid::

    flowchart LR
        gtb-->cls
        gtb["Ground Truth"]-->iou
        pdm-.->cls["Max Class"]
        pdm["Main Prediction"]-.->iou["Closest IoU"]
        pdm-.->anc["box in anchor"]
        cls-->gt
        iou-->gt["Matched GT Box"]
        anc-.->gt

        gt-->Liou["IoU Loss"]
        pdm-->Liou
        pdm-->Lbce
        gt-->Lbce["BCE Loss"]
        gt-->Ldfl["DFL Loss"]
        pdm-->Ldfl

        Lbce-->ML
        Liou-->ML
        Ldfl-->ML["Total Loss"]

- **Ground Truth**: The actual labels and bounding boxes in the dataset.
- **Main Prediction**: The model's predicted bounding boxes and class scores.
- **IoU (Intersection over Union)**: Measures the overlap between the predicted and ground truth boxes.
- **BCE (Binary Cross-Entropy) Loss**: Used for class prediction.
- **DFL (Distribution Focal Loss)**: Used for improving the precision of bounding box regression.

By optimizing these components, YOLOv9 aims to achieve high accuracy and robustness in object detection tasks.
