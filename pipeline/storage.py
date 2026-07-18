"""Local-disk output layout + (deferred) Google Cloud Storage upload.

Local disk is treated as the default sink; GCS is added as a second sink once a bucket
and credentials are configured (GCS_BUCKET + GOOGLE_APPLICATION_CREDENTIALS env vars) --
until then, upload calls are a documented no-op rather than failing the pipeline.

Local retention is capped: at ~170MB/file (int16-packed) and up to 161 files/forecast
cycle, one cycle is already ~25-30GB. This box has a 100GB disk, so without pruning a
handful of unattended cycles would fill it. We keep only the most recent N cycles locally
and rely on GCS (once configured) as the durable long-term store.
"""
import os
import shutil

OUTPUT_ROOT = "outputs"


def cycle_dir(init_time, root=OUTPUT_ROOT):
    path = os.path.join(root, str(init_time))
    os.makedirs(path, exist_ok=True)
    return path


def local_path(init_time, forecast_hour, root=OUTPUT_ROOT):
    return os.path.join(cycle_dir(init_time, root), f"wm3_f{forecast_hour:03d}.nc")


def prune_old_cycles(keep=2, root=OUTPUT_ROOT):
    """Deletes all but the `keep` most-recently-initialized cycle directories."""
    if not os.path.isdir(root):
        return []
    cycles = sorted(
        (d for d in os.listdir(root) if d.isdigit() and os.path.isdir(os.path.join(root, d))),
        key=int, reverse=True,
    )
    removed = []
    for d in cycles[keep:]:
        path = os.path.join(root, d)
        shutil.rmtree(path)
        removed.append(path)
    return removed


def gcs_enabled():
    return bool(os.environ.get("GCS_BUCKET"))


def upload_to_gcs(local_file_path, remote_blob_name=None):
    """No-op (logged) until GCS_BUCKET is set. Uses google-cloud-storage + the ambient
    credentials (GOOGLE_APPLICATION_CREDENTIALS service-account key, or
    `gcloud auth application-default login`)."""
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        print(f"[storage] GCS_BUCKET not set, skipping upload of {local_file_path}")
        return None

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = remote_blob_name or local_file_path
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_file_path)
    uri = f"gs://{bucket_name}/{blob_name}"
    print(f"[storage] Uploaded {local_file_path} -> {uri}")
    return uri
