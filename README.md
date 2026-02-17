# automatic-cuepoints
Software to automatically place cue points on tracks for easier mixing.

## Supported Models
### STFT Modal (Local)
Simple, fast local model that separates tracks into frequency bands and uses change point detection algorithms to find most likely segment boundaries.

Primary weakness right now is no understanding of song structure / repetition and no segment labeling. Keeping common annotations for similar section (e.g. two choruses) can significantly improve accuracy. The `recurrence_engine.py` is a WIP but is working on addressing these shortcomings.

### All-in-One (Remote via Modal)
Deployment of [all-in-one](https://github.com/mir-aidj/all-in-one) model for segmentation. Read more about this approach in the corresponding [research paper](https://arxiv.org/abs/2307.16425).

Because this model is quite compute intensive (demucs for stem splitting and NATTEN for neighborhood attention), we only support running this model remotely via Modal, which has a generous free plan ($30/month). You'll need to provide your `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the app to run this model.

Compute costs for each song is a few cents, so you should be able to easily label >1000 songs per month without paying more money.


## CLI Support

Run the CLI via `demo.py`:

```bash
python demo.py [OPTIONS]
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `calc-metrics` | Mode to run: `calc-metrics` or `add-cuepoints` |
| `--model` | `recurrence` | Model to use: `recurrence`, `stft`, or `all_in_one` |
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

## Rebuilding App
We use PyInstaller to build a simple Mac app (built with PyQt) for this program.
```
pyinstaller app.spec --noconfirm && mkdir dist/Autocuepoints.app/Contents/Frameworks/pyrekordbox
```

You can then find and run this app under: `automatic-cuepoints/dist/`.

## In-Progress Work
- Pre-download demucs model in `:cuda` Docker image to speed up cold starts.
- Rewrite allin1 repo with modern dependencies (e.g. new version of NATTEN) for better performance / maintainability.
- Add automations for republishing Docker images + app builds.
- Add speed / cost statistics for allin1 engine to add better estimates for number of songs that you can analyze per month (under Modal free plan).
