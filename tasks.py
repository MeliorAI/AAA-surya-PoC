import glob
import json
import os
from collections import defaultdict
from functools import partialmethod

from funcy import chunks
from invoke import task
from surya.input.load import load_pdf
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from surya.ocr import run_ocr
from tqdm import tqdm
from tqdm.rich import tqdm as pbar


@task(
    help={
        "input_dir": "Directory containing PDFs to OCR",
        "output_dir": "Directory to output the results",
        "langs": "comma separated list of Languages",
    }
)
def process_dir(
    ctx, input_dir: str, output_dir: str, langs: str = "en", batch_size: int = 4
):
    """Insipired by:
    https://github.com/VikParuchuri/surya/blob/master/ocr_text.py
    """
    langs = langs.split(",")

    pdfs = glob.glob(os.path.join(input_dir, "*.pdf"))
    print(f"📚️ Found {len(pdfs)} PDFs")

    if len(pdfs) < 1:
        return

    print(f"📥️ Loading models...")
    det_processor, det_model = segformer.load_processor(), segformer.load_model()
    rec_model, rec_processor = load_model(), load_processor()

    pdf_iter = pbar(pdfs, dynamic_ncols=True)

    # dissable any tqdm pbar from here onwards... Thx surya...
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    for pdf in pdf_iter:
        pdf_dir, fname = os.path.split(pdf)
        fname, _ = os.path.splitext(fname)
        doc_cls = os.path.split(pdf_dir)[-1]
        pdf_iter.set_description(f"⚙️ {fname} ({doc_cls})")

        # Check wether a results file for this doc exists
        results_dir = os.path.join(output_dir, doc_cls)
        results_file = os.path.join(results_dir, f"{fname}.json")
        if os.path.exists(results_file):
            continue

        # Extract pages
        images, names = load_pdf(pdf)
        pdf_iter.set_description(f"⚙️ {fname} ({doc_cls}) [{len(images)} pages]")

        # Run OCR
        predictions_by_image = []
        for im_batch in chunks(batch_size, images):
            pred_batch = run_ocr(
                im_batch, [langs], det_model, det_processor, rec_model, rec_processor
            )
            predictions_by_image.extend(pred_batch)

        # Compose output
        out_preds = defaultdict(list)
        for name, pred, _ in zip(names, predictions_by_image, images):
            out_pred = pred.model_dump()
            out_pred["page"] = len(out_preds[name]) + 1
            out_preds[name].append(out_pred)

        # Make the results dir if not present
        os.makedirs(results_dir, exist_ok=True)

        # Write as JSON file
        with open(results_file, "w+", encoding="utf-8") as f:
            json.dump(out_preds, f, ensure_ascii=False)
