import re
from robobrowser import RoboBrowser
import pickle

ids = ''

f = open('../movie_ids.txt')

for i in range(40000):
    movie_id = f.readline().replace('\n', '')

    try:
        browser = RoboBrowser(history=True, parser='html.parser', timeout=10)

        browser.open('http://www.imdb.com/title/'+movie_id)

        genre = str(browser.find(class_=re.compile(r'\bitemprop\b'))).split('>')[1].split('<')[0]
        print(i+1, movie_id, genre)

        poster_tag = str(browser.find(class_=re.compile(r'\bposter\b')))
        if poster_tag == "None":
            print('Error!')
            continue

        print('Loading .. ', end='')

        try:
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

                with open('../genres.txt', 'a') as l:
                    l.write(movie_id+','+genre+'\n')
                    print('Done')
        except:
            print("Fail!")
    except:
        print('Timeout')
