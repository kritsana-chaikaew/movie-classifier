import re
from robobrowser import RoboBrowser

movie_id = 'tt1825683'
browser = RoboBrowser(history=True, parser='html.parser')

browser.open('http://www.imdb.com/title/'+movie_id)
poster_tag = str(browser.find(class_=re.compile(r'\bposter\b')))

viewer_url = 'http://www.imdb.com/'+poster_tag.split('"')[3]
viewer_url = viewer_url.split("?")[0]
browser.open(viewer_url)

img_id = str(browser.find('link')).split('"')[1].split('/')[-1]
gallery = str(browser.find('script'))

start = gallery.find('"id":"' + img_id + '"')
if start != -1:
    start = gallery.find('"src"', start)
    if start != -1:
        end = gallery.find('"w"', start)
        img_url = gallery[start:end].split('"')[3]

print(img_url)
