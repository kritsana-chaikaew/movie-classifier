import re
from robobrowser import RoboBrowser
import pickle
import time
import os

def pad(num):
    s = str(num)
    return '000000'[0:6-len(s)] + s

f = open('../movie_ids.txt')
start_line = int(input('Start at line (inclusive): '));
end_line = int(input('End at line (inclusive): '));

path_name = pad(start_line) + '-' + pad(end_line)
temp_file = '../' + path_name + '.tmp'

if not os.path.exists("../posters"):
    os.makedirs("../posters")

if not os.path.exists("../genres-list"):
    os.makedirs("../genres-list")

countries = ['USA', 'UK', 'Thai']
counter = 0

for i, line in enumerate(f):
    movie_id = line.replace('\n', '')
    if i + 1 < start_line:
        continue
    if i + 1 > end_line:
        break

    retry_count = 0
    while True:
        retry_count += 1
        if retry_count >= 20:
            break
            
        print(i + 1, movie_id, end='\t')
        try:
            browser = RoboBrowser(history=True, parser='html.parser', timeout=10)

            browser.open('http://www.imdb.com/title/'+movie_id)
            poster_tag = str(browser.find(class_=re.compile(r'\bposter\b')))

            browser.select('id.titleDetails')
            country = str(browser.find(href=re.compile(r'\?country')).text)
            print(country, end='\t')

            browser.select('id.titleStoryLine')
            genres = browser.find_all(href=re.compile(r'\?ref_=tt_stry_gnr'))

            for j in range(len(genres)):
                genres[j] = str(genres[j]).split('> ')[1].split('<')[0]
            genres = str(genres).replace('[', '').replace(']', '')
            genres = genres.replace(', ', ':')
            genres = genres.replace('\'', '')

            print(genres)

            if country not in countries:
                print('Skip')
                break

            if poster_tag == "None":
                print('No Poster')
                break

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

                with open(temp_file, 'a') as g:
                    g.write(movie_id+','+genres+'\n')

                counter += 1
                print('Done')
            except:
                print("Fail!")
        except KeyboardInterrupt:
            exit()
        except:
            print('Timeout, Retrying')
            time.sleep(5)
            continue
        break

with open(temp_file, 'r') as duplicate:
    lines=duplicate.readlines()
    uniquelines=set(lines)

    with open('../genres-list/'+path_name+'.txt', 'w') as unique:
        unique.writelines(uniquelines)

try:
    os.remove(temp_file)
except:
    pass

print('Downloaded', counter, 'posters')
