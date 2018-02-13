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
Script will create a file named i-j.txt where i is a start line number and j is an end line number.
So, you can split data and download a chunk of it multiple times.
Recommneded size is 1000-2000 lines per file, downloading time is about an hour depends on connection speed.
