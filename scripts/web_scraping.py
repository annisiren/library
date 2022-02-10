##################
# To INSTALL
# pip install requests
# pip install lxml
# pip install bs4

import requests

# Step 1: Use the requests library to grab the page
# Note, this may fail if you have a firewall blocking Python/Jupyter
# Note sometimes you need to run this twice if it fails the first time
res = requests.get("http://www.example.com")

import bs4

soup = bs4.BeautifulSoup(res.text,"lxml")
