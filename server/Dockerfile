# doctranslate engine — build context is the REPO ROOT:
#   docker build -f server/Dockerfile -t doctranslate .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    HOME=/data

# ocrmypdf runtime deps + tesseract + fonts (Noto covers Arabic fallback rendering).
#
# tesseract-ocr-ara is not optional: a scanned page is very often MIXED — an
# English handout under an Arabic university header, an instructor's Arabic
# annotations — and tesseract reads Arabic it has no model for as Latin noise
# (`algo Gill`, `sow Josi`, `Pxé`), which the pipeline then translates and
# draws over the reader's own page. Naming a model that is not installed makes
# tesseract exit non-zero, so ocr_prep.tesseract_lang asks `--list-langs`
# first; that is what keeps this line and the code honest with each other.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-ara \
        ghostscript \
        pngquant unpaper qpdf \
        libjpeg62-turbo zlib1g \
        fonts-noto-core fonts-noto-ui-core \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# No office suite here. /v1/convert renders through the shared Gotenberg
# service over HTTP (GOTENBERG_URL, see server/convert.py) — a container that
# already carries LibreOffice for every app on the box. Baking a second copy in
# cost this image ~515 MB and this container a forked soffice per request, out
# of a 3 GB cap the translation pass wants for itself.

WORKDIR /app
COPY . /app

RUN pip install -e /app && pip install -r /app/server/requirements.txt

# Pre-download babeldoc assets (fonts + onnx layout model) into the image so
# first boot in prod does not depend on external downloads. babeldoc caches
# under $HOME/.cache/babeldoc; the entrypoint seeds /data from this bake.
RUN HOME=/opt/babeldoc-cache \
    python -c "import babeldoc.assets.assets as a; a.warmup()" \
    && chmod -R a+rX /opt/babeldoc-cache

# Seed /data cache from the baked assets at container start, then run the API.
COPY server/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME /data
EXPOSE 8000
CMD ["/entrypoint.sh"]
