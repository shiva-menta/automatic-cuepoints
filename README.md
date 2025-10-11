# automatic-cuepoints
Software to automatically place cue points on tracks for easier mixing.


## Building App
We use PyInstaller to build a simple Mac app for this program.

```
pipenv shell
pyinstaller app.spec --noconfirm && mkdir dist/Autocuepoints.app/Contents/Frameworks/pyrekordbox
```