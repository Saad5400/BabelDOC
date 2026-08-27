#!/usr/bin/env python
"""Full-deck integration run: image_prep -> babeldoc(+regions) -> mono+sidecar
-> side_by_side + interlinear. Mirrors server/pipeline.py's digital branch.

Usage: full_run.py input.pdf outdir
Env: OPENAI_API_KEY
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

input_pdf = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

import babeldoc.format.pdf.high_level as high_level
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.format.pdf.high_level import async_translate
from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
from babeldoc.glossary import Glossary
from babeldoc.translator.translator import set_translate_rate_limiter
from server.cost import CostTrackingTranslator
from server import compose, config, interlinear

prep_pdf = out_dir / "prep.pdf"
regions_json = out_dir / "regions.json"
subprocess.run([sys.executable, str(config.IMAGE_PREP_SCRIPT),
                str(input_pdf), str(prep_pdf), str(regions_json)], check=True)
regions = json.loads(regions_json.read_text())
use_prep = bool(regions.get("pages"))
print(f"image_prep: {sum(len(v) for v in regions.get('pages', {}).values())} regions "
      f"on {len(regions.get('pages', {}))} pages")

high_level.init()
model = DocLayoutModel.load_onnx()
translator = CostTrackingTranslator(
    lang_in="en", lang_out="ar", model=config.OPENAI_MODEL,
    base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
set_translate_rate_limiter(4)

glossaries = []
if config.GLOSSARY_PATH.is_file():
    g = Glossary.from_csv(config.GLOSSARY_PATH, "ar")
    if g.entries:
        glossaries.append(g)

sidecar_path = out_dir / "sidecar.json"
tc = TranslationConfig(
    translator=translator,
    input_file=str(prep_pdf if use_prep else input_pdf),
    lang_in="en", lang_out="ar",
    doc_layout_model=model, output_dir=str(out_dir),
    no_dual=True, no_mono=False,
    watermark_output_mode=WatermarkOutputMode.NoWatermark,
    glossaries=glossaries, auto_extract_glossary=False,
    translation_sidecar_path=sidecar_path,
    image_text_regions=regions_json if use_prep else None,
)

holder = {}
async def consume():
    async for event in async_translate(tc):
        t = event.get("type")
        if t == "finish":
            holder["result"] = event["translate_result"]
            break
        if t == "error":
            holder["error"] = str(event.get("error"))
            break

asyncio.run(consume())
if "error" in holder:
    sys.exit(f"babeldoc failed: {holder['error']}")

res = holder["result"]
mono = Path(res.no_watermark_mono_pdf_path or res.mono_pdf_path)
(out_dir / "mono.pdf").write_bytes(mono.read_bytes())

orig = input_pdf.read_bytes()
(out_dir / "side_by_side.pdf").write_bytes(
    compose.compose_dual(orig, (out_dir / "mono.pdf").read_bytes(), "side_by_side"))

sidecar = json.loads(sidecar_path.read_text())
inter, report = interlinear.render_overlay(orig, sidecar)
(out_dir / "interlinear.pdf").write_bytes(inter)
print("interlinear report:", report)

spend = translator.spend()
print("cost_usd:", spend["cost_usd"], "calls:", spend["calls"])
print("done ->", out_dir)
