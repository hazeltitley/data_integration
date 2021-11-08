# Data Integration and Web Scraping
Name: Hazel Titley

Date: 25/10/2021

Environment: Python 3.7.9 and Anaconda 2020.11

Libraries used:
* pandas
* numpy
* BeautifulSoup
* requests
* datetime
* statistics
* PyPDF2
* re
* ast
* math
* sklearn (specifically haversine_distance and preprocessing)
* matplotlib
* scipy
* shapely
* shapefile

This code aims to integrate data from multiple sources, to form a pool of property and covid data. The code is split into two main tasks. In Task 1, the suburb, LGA, closest train station, fastest route to Melbourne Central (MC), and covid data for each property must be deterined. To achieve this, the code is broken into multiple parts. Task 2 deals with exploration of the covid data, and using preceeding case averages to determine the case data for September 30th. The breakdown of the code is as follows:
Part 0:
- importing libraries and data <br>

Task 1:
- Part 1: adding data defaults
- Part 2: calculating suburb
- Part 3: filling in LGA
- Part 4: determining closest train station
- Part 5: determining min travel time to MC
- Part 6: web scraping to find covid data
- Part 7: fixing data to match sample output
- Part 8: exporting data to csv

Task 2:
- Part 1: data exploration
- Part 2: linear model building

Each part is outlined in more detail within each section.

Note: Some of the files need extracting before the code will run
