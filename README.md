# movie-classifier

development using python3

### Install Python and Virtual Environment
https://www.anaconda.com/download/ (Recommend for Windows)

or

https://virtualenv.pypa.io/en/stable/installation/


### Create Virtual Environment and Activate
```bash
conda create -n ENV pip python=3.6 
activate ENV
```
```bash
virtualenv -p python3 ENV
source ENV/bin/activate
```

### Crawling Data
```bash
$ pip install robobrowser
$ python crawler.py
Start at line (inclusive): 1
End at line (inclusive): 100
```
