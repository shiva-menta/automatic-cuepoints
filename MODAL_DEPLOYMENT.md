# Modal Deployment Guide for AllInOneEngine

## Prerequisites

1. Install Modal CLI:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

## Deployment Steps

### 1. Deploy the Modal App

Run the following command from the repository root:

```bash
modal deploy track_interface/cuepoint_engines/all_in_one_engine.py
```

This will:
- Build the Docker image with CUDA, PyTorch, allin1, madmom, and NATTEN
- Deploy the `find_segments` function as a web endpoint
- Output a URL like: `https://your-username--automatic-cuepoints-find-segments.modal.run`

### 2. Update the Endpoint URL

Copy the endpoint URL from the deployment output and update it in `demo.py`:

```python
case "all_in_one":
    params = AllInOneEngineParams(
        debug_mode=debug_mode,
        modal_endpoint_url="https://YOUR-ACTUAL-URL-HERE.modal.run"  # <- Paste here
    )
```

### 3. Run Your Demo

Now you can run your demo script with the all_in_one model:

```bash
python demo.py --model all_in_one --mode calc-metrics
```

## Testing the Endpoint

You can test the endpoint directly with curl:

```bash
curl -X POST https://your-endpoint-url.modal.run \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/path/to/your/file.wav
```

Expected response:
```json
{
  "segments": [0.0, 15.5, 30.2, 45.7, ...]
}
```

## Monitoring

- View logs: `modal app logs automatic-cuepoints`
- List deployments: `modal app list`
- Stop the app: `modal app stop automatic-cuepoints`

## Cost Considerations

Modal charges based on:
- **Compute time**: Only when the function is running
- **GPU usage**: The CUDA image uses GPU resources

The function will automatically scale to zero when not in use, so you only pay for actual usage.

## Troubleshooting

### Issue: "allin1 could not be resolved"
This is just a local IDE warning. The `allin1` package will be available in the Modal container.

### Issue: "modal_endpoint_url not set"
Make sure you've updated the endpoint URL in `demo.py` after deployment.

### Issue: CUDA/PyTorch version errors
Check the Modal image configuration matches the requirements from the all-in-one repository.
