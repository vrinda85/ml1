# Python script to scrape an article given the url of the article and store the extracted text in a file
# Url: https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7

import os
import requests
import re
url = ""
import sys
# Code here - Import BeautifulSoup library
from bs4 import BeautifulSoup
# Code ends here

# function to get the html source text of the medium article
def get_page():
	#global url
	url = input("Hello User , Please enter the URL of the data that needs to be written to file")
	
	# Code here - Ask the user to input "Enter url of a medium article: " and collect it in url
    #url = "https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7"
	# Code ends here
	
	# handling possible errorx	
	if not re.match(r'https?://medium.com/',url):
		print('Please enter a valid website, or make sure it is a medium article')
		sys.exit(1)

	# Code here - Call get method in requests object, pass url and collect it in res
	res = requests.get(url)
	# Code ends here

	res.raise_for_status()
	soup = BeautifulSoup(res.text, 'html.parser')
	return soup

# function to remove all the html tags and replace some with specific strings
def clean(text):
    rep = {"<br>": "\n", "<br/>": "\n", "<li>":  "\n"}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    text = re.sub('\<(.*?)\>', '', text)
    return text


def collect_text(soup):
	text = f'url: {url}\n\n'
	para_text = soup.find_all('p')
	print(f"paragraphs text = \n {para_text}")
	for para in para_text:
		text += f"{para.text}\n\n"
	return text

# function to save file in the current directory
def save_file(text):
	if not os.path.exists('./scraped_articles'):
		os.mkdir('./scraped_articles')
	name = url.split("/")[-1]
	print(name)
	fname = f'scraped_articles/{name}.txt'
	print(fname)
	print(f'File saved in directory {fname}')
	file1 = open(fname, 'w')
	file1.write(text)
	file1.close()
	

	



if __name__ == '__main__':
	url = "https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7"
	text = collect_text(get_page())
	save_file(text)
	# Instructions to Run this python code
	# Give url as https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7
	