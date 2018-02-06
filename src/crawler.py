import re
from robobrowser import RoboBrowser

ids = ''

f = open('../movie_ids.txt')

for i in range(100):
    movie_id = f.readline().replace('\n', '')
    print(i+1, movie_id)


    browser = RoboBrowser(history=True, parser='html.parser')

    browser.open('http://www.imdb.com/title/'+movie_id)
    poster_tag = str(browser.find(class_=re.compile(r'\bposter\b')))
    if poster_tag == "None":
        continue
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

    request = browser.session.get(img_url, stream=True)

    browser.open(img_url)
    with open('../posters/'+movie_id+'.jpg', 'wb') as j:
        j.write(request.content)
