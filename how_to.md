1. Custom Natten Wheels

```
docker build -t automatic-cuepoints .
docker run -it -v /path/on/your/mac:/mnt/local --name autocuepoints automatic-cuepoints
```

For Modal
```
pip3 install natten==0.17.4+torch250cu124 -f https://whl.natten.org/old
```

For Local Dev
```
pip3 install natten==0.17.4+torch250cpu -f https://whl.natten.org/old
```