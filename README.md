# automatic-cuepoints
Software to automatically place cue points on tracks for easier mixing.

## Prerequisites
- Rekordbox v6 — Rekordbox v7 makes it harder to access the local database key used for song data and cuepoints. If you install Rekordbox v6 alongside v7, the key is much easier to retrieve. You can download older versions
  [here](https://rekordbox.com/en/support/faq/v6/#faq-q600141). Installing v6 won't affect your v7 usage and you can continue to use the v7 Desktop App.
- Modal Account (optional) - We use Modal to run more powerful segmentation models (with label recognition) on GPUs in the cloud.

## Using the App
We provide a simple Mac app (built with PyQt) so you can use this tool without any coding. Download the latest release and open `Autocuepoints.app`.

To rebuild the app from source:
```
uv run pyinstaller app.spec --noconfirm && open dist/Autocuepoints.app
```

## Using the CLI 

Run the CLI via `demo.py`:

```bash
python demo.py [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `calc-metrics` | Mode to run: `calc-metrics` or `add-cuepoints` |
| `--model` | `cbm` | Model to use: `recurrence`, `cbm`, or `all_in_one` |
| `--num-processes` | `1` | Number of parallel processes |
| `--debug` | `False` | Enable debug logging and force single process |
| `--default-track` | `False` | Run on default test track only |
| `--num-songs` | `0` | Number of songs to process (0 = all) |

**Examples:**

```bash
# Calculate metrics using recurrence model
python demo.py --mode calc-metrics --model recurrence

# Add cuepoints to test_data playlist using all_in_one model
python demo.py --mode add-cuepoints --model all_in_one --num-processes 4

# Debug mode with default track
python demo.py --mode calc-metrics --debug --default-track
```

## Supported Models
### CBM Model (Local)
Correlation Block-Matching (CBM) algorithm for music structure segmentation. Based on the paper by Marmoret et al. (2023): "Barwise Music Structure Analysis with the Correlation Block Matching Segmentation Algorithm" ([DOI](https://doi.org/10.5334/tismir.167)).

This is the default local model - it's fast and provides good accuracy without requiring remote compute.

### All-in-One (Remote via Modal)
Deployment of [all-in-one](https://github.com/mir-aidj/all-in-one) model for segmentation. Read more about this approach in the corresponding [research paper](https://arxiv.org/abs/2307.16425).

Because this model is quite compute intensive (demucs for stem splitting and NATTEN for neighborhood attention), we only support running this model remotely via Modal, which has a generous free plan ($30/month). You'll need to provide your `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the app to run this model.

Compute costs for each song is a few cents, so you should be able to easily label >1000 songs per month without paying more money.

I am working on modernizing this model for faster and cheaper performance [here](https://github.com/shiva-menta/all-in-one-modernized).

## Deploying Images & Serverless Functions
### allin1 Dockerfile
If any changes are made to dependencies for running allin1 model via Dockerfile on Modal, you'll need to rebuild / republish the image to DockerHub. Modal uses this published image to run the segmentation model.
```
docker buildx build --platform linux/amd64 -f cuda.Dockerfile -t smenta/automatic-cuepoints:cuda --push .
```

### Modal Serverless Function
If any changes are made in `track_interface/cuepoint_engines/modal_app:process_audio` you'll need to redeploy the Modal serverless function so any updates are propagated to Modal.
```
modal deploy track_interface/cuepoint_engines/modal_app.py
```

## In-Progress Work
- Pre-download demucs model in `:cuda` Docker image to speed up cold starts.
- Rewrite allin1 repo with modern dependencies (e.g. new version of NATTEN) for better performance / maintainability.
- Add automations for republishing Docker images + app builds.
- Add speed / cost statistics for allin1 engine to add better estimates for number of songs that you can analyze per month (under Modal free plan).
