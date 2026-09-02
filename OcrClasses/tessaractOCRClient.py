from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class TesseractOCRClient:
    """Free, local, offline. Best for clean printed text/scans."""

    def extract_text(self, file_path: str) -> str:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)