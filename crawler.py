import re
from robobrowser import RoboBrowser

movie_id = 'tt1825683'
browser = RoboBrowser(history=True, parser='html.parser')
browser.open('http://www.imdb.com/title/'+movie_id)

poster = str(browser.find(class_=re.compile(r'\bposter\b')))

viewer_url = 'http://www.imdb.com/'+poster.split('"')[3]
print(viewer_url)
browser.open(viewer_url)
