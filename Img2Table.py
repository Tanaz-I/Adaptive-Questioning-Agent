import os
os.environ["PATH"] += r";C:\Program Files\Tesseract-OCR"

def extract_table_from_image(image_bytes):
    from img2table.document import Image
    img = Image(src=image_bytes)

    from img2table.ocr import PaddleOCR
    paddle_ocr = PaddleOCR(lang="en")

    from img2table.ocr import EasyOCR

    class MyEasyOCR(EasyOCR):
        def __init__(self, lang = None, kw = None) -> None:
            super().__init__(lang=lang, kw=kw)

        def content(self, document) -> list[list[tuple]]:
            return [self.reader.readtext(image, min_size=5,
                                        contrast_ths=0.05,
                                        adjust_contrast=0.7,
                                        text_threshold=0.3,
                                        low_text=0.2,
                                        link_threshold=0.2
                                        ) for image in document.images]


    easyocr = MyEasyOCR(lang=["en"], kw={"gpu": False})

    extracted_tables = img.extract_tables(ocr=easyocr,
                                      implicit_rows=False,
                                      borderless_tables=False,
                                      min_confidence=0)

    print("RAW OUTPUT:\n", extracted_tables[0].df)

    return extracted_tables


with open("zz.png", "rb") as f:
    image_bytes = f.read()

    table_text = extract_table_from_image(image_bytes)
    processed_text = table_text
