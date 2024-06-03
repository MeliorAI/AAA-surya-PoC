import glob
import json
import os
from collections import defaultdict
from functools import partialmethod
from typing import List

from funcy import chunks
from surya.input.load import load_pdf
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from surya.ocr import run_ocr
from tqdm import tqdm
from tqdm.rich import tqdm as tqdm_pbar

from src.surya_exploratory import catchtime


class OCR:
    def __init__(self) -> None:
        print(f"📥️ Loading models...")
        self.det_processor = segformer.load_processor()
        self.det_model = segformer.load_model()
        self.rec_model = load_model()
        self.rec_processor = load_processor()

    def ocr_dir(
        self,
        input_dir: str,
        output_dir: str,
        langs: str = "en",
        batch_size: int = 4,
        time_profile: bool = False,
    ):
        """Apply OCR to each PDF in the given directory

        Args:
            input_dir (str): _description_
            output_dir (str): _description_
            langs (str, optional): _description_. Defaults to "en".
            batch_size (int, optional): _description_. Defaults to 4.
        """
        pdfs = glob.glob(os.path.join(input_dir, "*.pdf"))
        print(f"📚️ Found {len(pdfs)} PDFs")

        if len(pdfs) < 1:
            return

        pbar = tqdm_pbar(pdfs, dynamic_ncols=True)

        # dissable any tqdm pbar from here onwards... Thx surya...
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        avg_proc_time = 0
        total_pages = 0
        for pdf in pbar:
            pdf_dir, fname = os.path.split(pdf)
            fname, _ = os.path.splitext(fname)
            doc_cls = os.path.split(pdf_dir)[-1]
            pbar.set_description(f"⚙️ {fname} ({doc_cls})")

            # Check wether a results file for this doc exists
            results_dir = os.path.join(output_dir, doc_cls)
            results_file = os.path.join(results_dir, f"{fname}.json")
            if os.path.exists(results_file):
                continue

            if time_profile:
                with catchtime() as t:
                    out_preds = self.ocr_pdf(pdf, langs, batch_size, pbar)

                print(
                    f"{fname}: {t.readout} [~{t.time/len(out_preds):.3f}s per page]"
                )

            # Make the results dir if not present
            os.makedirs(results_dir, exist_ok=True)

            # Write as JSON file
            with open(results_file, "w+", encoding="utf-8") as f:
                json.dump(out_preds, f, ensure_ascii=False)

    def ocr_pdf(
        self,
        pdf: str,
        langs: List[str],
        batch_size: int = 4,
        pbar: tqdm_pbar | None = None,
    ):
        """Applies OCR on the given PDF file

        Args:
            pdf (str): PDF file to OCR
        """
        # Extract pages
        images, _ = load_pdf(pdf)

        _, fname = os.path.split(pdf)
        msg = f"⚙️ {fname} [{len(images)} pages]"
        if pbar:
            pbar.set_description(msg)
        else:
            print(msg)

        # Run OCR
        predictions_by_image = []
        for im_batch in chunks(batch_size, images):
            pred_batch = run_ocr(
                im_batch,
                [langs],
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor,
            )
            predictions_by_image.extend(pred_batch)

        # Compose output
        out_preds = []
        for pn, (pred, _) in enumerate(zip(predictions_by_image, images)):
            out_preds.append(
                {
                    "ocr": pred.model_dump(),
                    "page": pn + 1,
                }
            )

        return out_preds
