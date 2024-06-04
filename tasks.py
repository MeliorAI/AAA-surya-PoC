import json

from invoke import task

from src.surya_exploratory import catchtime
from src.surya_exploratory.ocr import OCR


@task(
    help={
        "input_dir": "Directory containing PDFs to OCR",
        "output_dir": "Directory to output the results",
        "langs": "comma separated list of Languages",
    }
)
def process_dir(
    ctx,
    input_dir: str,
    output_dir: str,
    langs: str = "en",
    batch_size: int = 4,
    time_profile: bool = False,
):
    """Insipired by:
    https://github.com/VikParuchuri/surya/blob/master/ocr_text.py
    """
    langs = langs.split(",")
    ocr = OCR()
    ocr.ocr_dir(input_dir, output_dir, langs, batch_size, time_profile)


@task
def process_pdf(
    ctx,
    input_pdf: str,
    output_json: str,
    langs: str = "en",
    batch_size: int = 4,
    time_profile: bool = False,
):
    langs = langs.split(",")
    ocr = OCR()

    with catchtime() as t:
        res = ocr.ocr_pdf(input_pdf, langs, batch_size)

    if time_profile:
        print(f"{t.readout} [~{t.time/len(res):.3f}s per page]")

    # Write as JSON file
    with open(output_json, "w+", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
