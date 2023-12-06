
from icrawler.builtin import GoogleImageCrawler
import os
os.chdir('C:\python_work08052021') 
while 1:
    google_crawler = GoogleImageCrawler(storage={'root_dir': 'trainimages3'})
    google_crawler.crawl(keyword='~car "thermal image"', max_num=900)