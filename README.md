# automatic-cuepoints
Software to automatically place cue points on tracks for easier mixing.


## Building App
We use PyInstaller to build a simple Mac app for this program.

```
pipenv shell
pyinstaller app.spec --noconfirm && mkdir dist/Autocuepoints.app/Contents/Frameworks/pyrekordbox
```

Current Status:
- Trying to hack a small one fix solution to the allin1 dependency issues.
- Simultaneously, use CC to fix dependencies to completely modern version of the other repository.