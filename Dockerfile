# Matches the exact torch/CUDA combo the pinned NATTEN wheel needs (torch 2.4.0 + cu121)
# -- see requirements/requirements-natten.txt. Do NOT bump torch (e.g. via
# requirements/requirements.txt's torch==2.5.1 pin); that breaks the NATTEN wheel tag.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl cron libgeos-dev libproj-dev proj-bin \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
        numpy xarray netCDF4 cartopy matplotlib cfgrib ecmwflibs \
        requests gdown google-cloud-storage \
    && pip install --no-cache-dir --index-url https://test.pypi.org/simple/ matepoint \
    && PYVER=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')") \
    && curl -sk -o /tmp/natten.whl \
        "https://shi-labs.com/natten/wheels/cu121/torch2.4.0/natten-0.17.3%2Btorch240cu121-${PYVER}-${PYVER}-linux_x86_64.whl" \
    && pip install --no-cache-dir /tmp/natten.whl && rm /tmp/natten.whl

ENV PYTHONPATH=/app

# Not baked into the image: weights/WeatherMesh3.pt (~1.2GB, licensed separately) and
# runtime output/log/cache dirs. Mount these at `docker run` time, e.g.:
#   docker run --gpus all \
#     -v /path/to/WeatherMesh3.pt:/app/weights/WeatherMesh3.pt \
#     -v $(pwd)/outputs:/app/outputs -v $(pwd)/logs:/app/logs \
#     -v $(pwd)/nomads_cache:/app/nomads_cache \
#     weathermesh3-pipeline
VOLUME ["/app/weights", "/app/outputs", "/app/nomads_cache", "/app/logs"]

CMD ["python3", "scripts/run_live_rollout.py"]
