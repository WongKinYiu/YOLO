from PIL import Image, ImageDraw, ImageFont


def draw_bboxes(img, bboxes):
    """
    Draw bounding boxes on an image.

    Args:
    - image_path (str): Path to the image file.
    - bboxes (list of lists/tuples): Bounding boxes with [x_min, y_min, x_max, y_max, class_id].
    """
    # Load an image
    draw = ImageDraw.Draw(img)

    # Font for class_id (optional)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    width, height = img.size

    for bbox in bboxes:
        class_id, x_min, y_min, x_max, y_max = bbox
        x_min = x_min * width
        x_max = x_max * width
        y_min = y_min * height
        y_max = y_max * height
        shape = [(x_min, y_min), (x_max, y_max)]
        draw.rectangle(shape, outline="red", width=2)
        draw.text((x_min, y_min), str(class_id), font=font, fill="blue")

    img.save("output.jpg")
