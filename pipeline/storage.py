"""Local-disk output layout + (deferred) S3 / Google Cloud Storage upload.

Local disk is treated as the default sink; S3 and/or GCS are added as second sinks once a
bucket and credentials are configured (S3_BUCKET / GCS_BUCKET env vars) -- until then,
upload calls are a documented no-op rather than failing the pipeline.

Local retention is capped: at ~170MB/file (int16-packed) and up to 161 files/forecast
cycle, one cycle is already ~25-30GB. This box has a 100GB disk, so without pruning a
handful of unattended cycles would fill it. We keep only the most recent N cycles locally
and rely on GCS (once configured) as the durable long-term store.
"""
import os
import re
import shutil
from datetime import datetime, timezone

OUTPUT_ROOT = "outputs"
_CYCLE_LABEL_RE = re.compile(r"^\d{8}_\d{2}z$")


def cycle_label(init_time):
    """Human-readable cycle id, e.g. 1784376000 -> '20260718_12z' (matches NOMADS' own
    gfs.<date>/<hour> convention, instead of an opaque unix timestamp in dir/key names)."""
    return datetime.fromtimestamp(init_time, tz=timezone.utc).strftime("%Y%m%d_%Hz")


def cycle_dir(init_time, root=OUTPUT_ROOT):
    path = os.path.join(root, cycle_label(init_time))
    os.makedirs(path, exist_ok=True)
    return path


def local_path(init_time, forecast_hour, root=OUTPUT_ROOT):
    return os.path.join(cycle_dir(init_time, root), f"wm3_f{forecast_hour:03d}.nc")


def prune_old_cycles(keep=2, root=OUTPUT_ROOT):
    """Deletes all but the `keep` most-recently-initialized cycle directories.
    'YYYYMMDD_HHz' sorts chronologically as a plain string, so no parsing needed."""
    if not os.path.isdir(root):
        return []
    cycles = sorted(
        (d for d in os.listdir(root) if _CYCLE_LABEL_RE.match(d) and os.path.isdir(os.path.join(root, d))),
        reverse=True,
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


def s3_enabled():
    return bool(os.environ.get("S3_BUCKET"))


def upload_to_s3(local_file_path, remote_key=None):
    """No-op (logged) until S3_BUCKET is set. Uses boto3's standard credential chain
    (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars, ~/.aws/credentials, or an IAM
    role) -- a *public* bucket almost always still means public-read, not anonymous
    write, so uploading still needs real AWS credentials even if the objects end up
    publicly readable once there."""
    bucket_name = os.environ.get("S3_BUCKET")
    if not bucket_name:
        print(f"[storage] S3_BUCKET not set, skipping upload of {local_file_path}")
        return None

    import boto3

    client = boto3.client("s3")
    key = remote_key or local_file_path
    client.upload_file(local_file_path, bucket_name, key)
    uri = f"s3://{bucket_name}/{key}"
    print(f"[storage] Uploaded {local_file_path} -> {uri}")
    return uri
