"""Small image-size helpers shared by the data pipeline and ROI code.

Kept in their own module so that both :mod:`frcnn.data` and :mod:`frcnn.roi`
can use them without creating a circular import.
"""


def get_img_output_length(width, height):
    """Feature-map size for the VGG-16 base (four 2x pooling stages => stride 16)."""
    def get_output_length(input_length):
        return input_length // 16

    return get_output_length(width), get_output_length(height)


def get_new_img_size(width, height, img_min_side=300):
    """Resize the smallest side of the image to ``img_min_side`` (keeping aspect)."""
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height
