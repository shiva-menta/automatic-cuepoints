import modal

# Modal App Configs
app = modal.App("automatic-cuepoints")
vol = modal.Volume.from_name("cache", create_if_missing=True)
cache_mount_path = "/root/cache"

# Volume filepaths
_OUT_DIR = f"{cache_mount_path}/analyze_outputs/"
_DEMIX_DIR = f"{cache_mount_path}/demix_outputs/"
_SPEC_DIR = f"{cache_mount_path}/spec_outputs/"

allin1_image = modal.Image.from_registry(
    "smenta/automatic-cuepoints:cuda",
)


@app.function(image=allin1_image, gpu="L40S", volumes={cache_mount_path: vol})
def process_audio(audio_bytes: bytes, file_name: str, force_recalculate: bool = False) -> list:
    """Internal function to process audio and return segments."""
    import allin1
    import os

    # write audio bytes to file
    tmp_path = f"/tmp/{file_name}"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    os.environ["TORCH_HOME"] = cache_mount_path

    # get analyze outputs
    result = allin1.analyze(
        tmp_path,
        out_dir=_OUT_DIR,
        demix_dir=_DEMIX_DIR,
        spec_dir=_SPEC_DIR,
        overwrite=force_recalculate,
        keep_byproducts=True,
    )
    segments = result.segments
    simple_segments = [{
        "start": float(segment.start),
        "end": float(segment.end),
        "label": segment.label,
    } for segment in segments]

    # todo(smenta) - figure out if we want to clear out the actual song data in file
    # modal doesn't charge for volume storage yet, but once this is set - we can remove.

    # persist changes
    vol.commit()

    return simple_segments
