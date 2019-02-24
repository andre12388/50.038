import requests

url = ('https://newsapi.org/v2/top-headlines?'
       'country=cn&'
       'apiKey=23ffcd607a0845b2a909a727aff6adbd')
response = requests.get(url)
print (response.json())

import json
with open('recent_china_news.json', 'w+') as outfile:
    json.dump(response.json(), outfile)
