from io import BytesIO

from PIL import Image


def load_document_as_rgb_images(file_bytes: bytes) -> list[Image.Image]:
    """
    Load an uploaded document as one or more RGB PIL images.

    - Regular image uploads are opened directly with Pillow.
    - PDF uploads are converted to one image per page via pdf2image.
    """
    if file_bytes.lstrip().startswith(b"%PDF"):
        from pdf2image import convert_from_bytes

        pages = convert_from_bytes(file_bytes)
        if not pages:
            raise ValueError("PDF has no renderable pages")
        return [page.convert("RGB") for page in pages]

    return [Image.open(BytesIO(file_bytes)).convert("RGB")]
