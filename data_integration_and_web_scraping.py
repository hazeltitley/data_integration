#!/usr/bin/env python
# coding: utf-8

# # Data Integration and Web Scraping
# Name: Hazel Titley
# 
# Date: 25/10/2021
# 
# Environment: Python 3.7.9 and Anaconda 2020.11
# 
# Libraries used:
# * pandas
# * numpy
# * BeautifulSoup
# * requests
# * datetime
# * statistics
# * PyPDF2
# * re
# * ast
# * math
# * sklearn (specifically haversine_distance and preprocessing)
# * matplotlib
# * scipy
# * shapely
# * shapefile
# 
# This code aims to integrate data from multiple sources, to form a pool of property and covid data. The code is split into two main tasks. In Task 1, the suburb, LGA, closest train station, fastest route to Melbourne Central (MC), and covid data for each property must be deterined. To achieve this, the code is broken into multiple parts. Task 2 deals with exploration of the covid data, and using preceeding case averages to determine the case data for September 30th. The breakdown of the code is as follows: <br>
# Part 0:
# - importing libraries and data <br>
# <br>
# Task 1:
# - Part 1: adding data defaults
# - Part 2: calculating suburb
# - Part 3: filling in LGA
# - Part 4: determining closest train station
# - Part 5: determining min travel time to MC
# - Part 6: web scraping to find covid data
# - Part 7: fixing data to match sample output
# - Part 8: exporting data to csv <br>
# <br>
# Task 2:
# - Part 1: data exploration
# - Part 2: linear model building
# 
# Each part is outlined in more detail within each section.
# <br>
# Note: Please ensure that the data is stored in the same directory as this jupyter file, otherwise the code will not run.

# ## Part 0 - Importing libraries and data
# before any data can be processed, it needs to read in to the working memory, and the relevant libraries imported. the base datasets (consisting of property locations and ids) are concatenated and duplicates removed to form a cleaned dataset. this data forms the basis of all later computations. that is, it is these properties for which the suburb, train station, and covid data is found. for this reason, it is important to ensure all properties are captured, as missing any values at this point would result in more missing data later on.

# In[1]:


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import datetime
import statistics
import PyPDF2 
import re
import ast
import math
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sklearn import preprocessing
from scipy import stats
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# the pandas function read_json can be used to read in the json file, with no further steps required. this results in a dataframe containing the data stored in the json file.

# In[2]:


content = pd.read_json('house_data.json')
print(content)


# reading in the data stored in the xml file is more complicated. this is partly because each property is stored as its own xml snippet. that is, theire are no enclosing tags for the whole dataset. to get around this, a tag "\<properties>" is used to wrap the data after reading it in. beautiful soup can then be used to convert it to xml format (Preste, 2019), and find_all called to find the relevant data (Shopeju, 2019). the text can then be extracted from this data and iterated through to produce a dataframe. this is then concatenated with the json data to produce a large dataframe containing all the basic property data.

# In[3]:


#initialising list
xml = []
#opening file
with open("house_data.xml") as file:
    #reading line by line
    xml = file.readlines()
    # joining lines of data
    xml = "".join(xml)
    #adding wrapper tag
    fixed = "<properties>" + xml + "</properties>"

#converting data to xml using beautifulsoup
soup = BeautifulSoup(fixed, 'xml')

#finding all useful data (Shopeju, 2019)
properties = soup.find_all('property_id')
lat = soup.find_all('lat')
lng = soup.find_all('lng')
addr_street = soup.find_all('addr_street')

#initialising data frame
data = pd.DataFrame()
for i in range(0, len(properties)):
    #extracting text from relevant tags and appending data to dataframe (Preste, 2019)
    new_data = pd.DataFrame({'property_id': properties[i].get_text(), 'lat': lat[i].get_text(), 'lng': lng[i].get_text(), "addr_street": addr_street[i].get_text()}, index=[0])
    data = pd.concat([data, new_data], ignore_index = True)

#combining json and xml data into one dataframe
combined_data = pd.concat([content, data], ignore_index = True)
#checking combination successful
combined_data.tail()


# as can be seen, the addition of the xml data to the json data hs been successful. however its possible this data now contains duplicates. these need to be removed before processing continues.

# In[4]:


#finding duplicates and printing them
pd.concat(g for _, g in combined_data.groupby("property_id") if len(g) > 1)


# as can be seen, the dataset does in fact contain perfect duplicates. these duplicates need to be removed, so that only one instance of each property exists

# In[5]:


#removing duplicates but keeping first instance
clean_data = combined_data.drop_duplicates(keep='first')
#double checking for any remaining duplicates
clean_data.duplicated().unique()


# as can be seen by the output, no duplicates remain.

# In[6]:


#resetting the index
clean_data.reset_index(drop=True, inplace=True)


# In[7]:


#reading remaining files
sf = shapefile.Reader("VIC_LOCALITY_POLYGON_shp")
stops = pd.read_csv('stops.txt')
stop_times = pd.read_csv('stop_times.txt')
trips = pd.read_csv('trips.txt')
calendar = pd.read_csv('calendar.txt')


# # Task 1:
# in this task, the suburb, LGA, closest train station, fastest route to Melbourne Central (MC), and covid data for each property is determined.

# ## Part 1 - adding data defaults
# to ensure correct structure of the dataframe, completion of task 1 is begun by adding all the requisite columns to the clean_data dataset. these columns are given the default values specified in the assignment outline.

# In[8]:


#creating required columns and setting default values
clean_data.loc[:, "suburb"] = "not available"
clean_data.loc[:, "Lga"] = "not available"
clean_data.loc[:, "closest_train_station_id"] = "0"
clean_data.loc[:, "distance_to_closest_train_station"] = "0"
clean_data.loc[:, "travel_min_to_MC"] = "-1"
clean_data.loc[:, "direct_journey_flag"] = "-1"
clean_data.loc[:, "30_sep_cases"] = "not available"
clean_data.loc[:, "last_14_days_cases"] = "not available"
clean_data.loc[:, "last_30_days_cases"] = "not available"
clean_data.loc[:, "last_60_days_cases"] = "not available"

#checking column creation has been successful
clean_data.head()


# as can be seen, all the required columns have been created, and their values successfully set to the default.

# ## Part 2 - calculating suburb
# the first value that needs to be calculated for each property is the suburb. as this is required in the caluclation of other columns. the suburb data has been given as a shapefile, which was previously read in as sf. to begin with, the shapes and records are extracted. to determine each properties corresponding suburb, a dictionary of suburbs is first created. the latitude and longittude of each property can then be converted into a shapefile point, and the dictionary looped through to determine if any suburb contains this point (Harris, 2013). the looping through the dictionary is defined as a function, so that it can be called for each property.

# In[9]:


#extracting records and shapes from shapefile data
recs = sf.records()
shapes = sf.shapes()


# In[10]:


#initialising dictionary
suburb_polygons = {}
for i, record in enumerate(recs):
    # the suburb is the 6th item in the list for each record, hence this indexing
    suburb = record[6]
    #extracting shape data
    points = shapes[i].points
    #converting to polygon
    poly = Polygon(points)
    #adding polygon to dictionary
    suburb_polygons[suburb] = poly


# In[11]:


# defining a function which checks if point a is in a suburb
# if it is, the suburb is returned, if not "not available" is returned
def in_suburb(lat, lon):
    p = Point(lon, lat)
    for suburb, poly in suburb_polygons.items():
        if poly.contains(p): # (Harris, 2013)
            return suburb
    return "not available"


# In[12]:


# converting data to numeric
clean_data.loc[:,"lat"] = pd.to_numeric(clean_data.loc[:,"lat"])
clean_data.loc[:,"lng"] = pd.to_numeric(clean_data.loc[:,"lng"])

#looping through data
for i in range(len(clean_data)):
    #exctrating latitude and longitude
    lon = clean_data.loc[i, "lng"]
    lat = clean_data.loc[i, "lat"]
    #calling function
    sub = in_suburb(lat,lon)
    #assigning reutrned value to dataframe
    clean_data.loc[i, "suburb"] = sub

#printing head to check if process has been successful
clean_data.head()


# as can be seen from the dataframe output, filling in of the suburb values has been successful.

# ## Part 3 - determining LGA
# now the suburb has been found, the LGA of each property can be determined. the LGA data is contained in a pdf, so this needs to be read in first. this is done using PyPDF2 (Phaseit Inc, 2021). each page of the PDF is read in, and using regex to find the LGA and associated suburbs (more informatioin below). newline characters need to be removed from the suburb names to avoid corruption of the data. to check the data has been read in correctly, the number of LGAs read in and the number of sets of suburbs must match. if these values do not match, and error message is printed. if no error arises, the data is saved to a dictionary with the LGA as the key, and the suburbs as values in a list. <br>
# N.B. regex: as the LGAs are saved in all caps, and preceed a ":". these values are used. the regex created specifies that the extracted string must begin with at least 1 capital letter, can contain anything (any number of times), and must finish with a capital, space, then colon. only the letters are captured, these correspond to the LGA. the suburb list is extracted similarly. all suburb data within the PDF is contained within square brackets and may consist of a mixture of characters (letter, or newline). hence, anything within square brackets is captured. this is done lazily, to avoid the whole PDF being captured as one set of suburbs. the regex was developed using https://regex101.com/ (Dib, 2021)

# In[13]:


#opening file
pdfFileObj = open('lga_to_suburb.pdf', 'rb') 
#creating a pdf reader object 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 

#initialising dictionary
LGA_dict = {}
#lopping through pdf pages
for page in range(pdfReader.numPages):
    pageObj = pdfReader.getPage(page) 
    #extracting text
    text = pageObj.extractText()
    #finding LGA and subrubs using regex
    LGA = re.findall(r'([A-Z]+.*[A-Z]) ?:',text)
    suburbs = re.findall(r"\[(.*?\n*?\r*?.*?\n*?.*)\]",text)
    #initialising list of suburbs
    fixed_sub = []
    for suburb in suburbs:
        #removing newline characters from text
        fixed = suburb.replace("\n", "")
        #appending fixed suburb names to list
        fixed_sub.append(fixed)
    # if the number of sets captured match, its safe to assume all the data has been captured
    if len(LGA) == len(suburbs):
        for i in range(len(LGA)):
            #saving LGA and suburbs to dictionary
            LGA_dict[LGA[i]] = ast.literal_eval(fixed_sub[i])
    # if the number of sets captured doesnt match, at least one error has occured
    else:
        #printing error message
        print("ERROR: PDF READ IN INCORRECTLY")

# closing the file
pdfFileObj.close()

#printing dictionary for inspection
print(LGA_dict)


# as can be seen, the dictionary has been created successfully. and so, can be used to fill in the LGA values in the clean_data dataset

# In[14]:


#looping through data
for i in range(len(clean_data)):
    #extracting suburb
    suburb = clean_data.loc[i, "suburb"]
    #looking for value in LGA_dict
    for key, value in LGA_dict.items():
        for v in value:
            if suburb and v and suburb.lower() == v.lower():
                #setting LGA to matching dict key once suburb found
                clean_data.loc[i, "Lga"] = key
#printing data fro inspection
clean_data.head()


# When inspecting the dataframe, I noticed that the LGA "strathbogie" was read in as "trathbogie". this may be caused by the regexing not being broad enough to capture all the characters effectively, or could be an error in reading in the pdf initially. either way, this needs fixing before processing continues. 

# In[15]:


#fixing dataframe values
for i in range(len(clean_data)):
    if clean_data.loc[i, 'Lga'] == "TRATHBOGIE":
        clean_data.loc[i, 'Lga'] = "STRATHBOGIE"


# if this process has been done properly, there should be minimal default values in the dataframe. for LGA, these are indicated by "not available". 

# In[16]:


print(clean_data.loc[clean_data['Lga'] == "not available"])


# as can be seen, only a few lines contain default data. this is a promising indicator.

# ## Part 4 - calculating the closest train station distance and id
# to determine which train station is closest to each property, the latitude and longitude of the property are first extracted from the dataframe. these are converted to radians, and used to calculate the haversine distance (Pedregosa et al., 2011) between the property and every station. the closest train station is found by updating the caluclated minium distance each time a smaller distance is found. the id of this station is also saved. once all the stations have been looped through, the value saved as the minimum distacne must correspond to the closest train station. the values in the clean_data dataframe are then filled changed to match these calculations.

# In[17]:


#getting latitude and longitude
for i in range(len(clean_data)):
    #extracting data
    lon = clean_data.loc[i, "lng"]
    lat = clean_data.loc[i, "lat"]
    min_dist = 1
    dist_changed = False
    #getting train station data
    for j in range(len(stops)):
        train_lat = stops.loc[j, "stop_lat"]
        train_lng = stops.loc[j, "stop_lon"]
        #converting to radians
        loc_rad = [radians(_) for _ in [lat,lon]]
        train_rad = [radians(_) for _ in [train_lat,train_lng]]
        #calculating haversine distance (Pedregosa et al., 2011)
        result = haversine_distances([loc_rad, train_rad])
        #if this distance is shorter than previously calculated distances, its saved
        result_dist = (result[0,1]+result[1,0])/2
        if result_dist<min_dist:
            min_dist = result_dist
            dist_changed = True
            closest_id = stops.loc[j, "stop_id"]
    #multiplying by earths radius to get km
    dist = min_dist * 6378
    if dist_changed:
        clean_data.loc[i, "closest_train_station_id"] = closest_id
        clean_data.loc[i, "distance_to_closest_train_station"] = dist

clean_data.head()


# as can be seen, the data has been calculated, and the values filled in.

# ## Part 5 - calculating minimum travel time to MC
# in this part, the average travel time for trips to melbourne central (MC) on weeekdays between 7 and 9am is calculated. to achieve this, a set of trips is first created. this set contains all trips which include MC as a stop and ran on all weekdays (as specified by the assignment FAQs). for each closest stop in the clean_data set, the set of trips is looped through. if the trip stops at the station, the trip is traced via its sequence number. this trip is followed until the train arrives at MC or the end of the trip is reached (indicated by a drop in sequence number). if MC is reached, the trip duration is added to a running total, and the number of trips incremented by one. once all the trips have been looped through, if the number of trips found is more than one, the average trip time is found by dividing the time_total by the number fo trips and this value saved to a dictionary with the train station id as the key. if no trips have been found, the value "not available" is saved with the station id as the key. <br>
# as this part of the code is the slowest, its important to optimise it to improve speed. this is first achieved by shrinking the dataset of trips. ths set intially consists of 23809 individual trips. by excluding all the trips which occured on a weekend (as identified using the corresponding service id and calender file), the set is shrunk to 13017 trips. by excluding those trips which dont have MC as a stop, the set is further reduced to 5963 trips. even with this filtering, the processing is still too slow without further ajustments. to bring the code up to sufficient speed, the data is converted from a pandas dataframe to a numpy array. this is much faster to process, as numpy preallocates memory. <br>
# the whole proces is split into three parts:
# - creation of the trip set
# - calculation of the shortest trip for each stop
# - filling in of the dataframe
# these steps are outlined below, as they are executed.

# ### 5.1 - creation of the trip set
# a list of services which run on weekdays is created. this is done by summing the weekday boolean values in the calendar file. if the sum is greater than 0, the service must run on at least one weekday.

# In[18]:


#initialising list
service_list = list()
#looping through services
for i in range(len(calendar)):
    #extracting id and weekday sum
    service_id = calendar.loc[i, "service_id"]
    weekday = calendar.iloc[i, 1]+calendar.iloc[i, 2]+calendar.iloc[i, 3]+calendar.iloc[i, 4]+calendar.iloc[i, 5]
    #appending service to list if it runs all weekdays
    if weekday==5:
        service_list.append(service_id)


# now a list of services has been created, the corresponding trips can be added to a set (this is done to avoid duplicates).

# In[19]:


#initialising set
trip_set = set()
for i in range(len(trips)):
    s_id = trips.loc[i, "service_id"]
    #adding trip to set if it corresponds to a service in service_list
    if (s_id in service_list):
        trip_set.add(trips.loc[i, "trip_id"])


# to reduce the number of trips further, a new set is created. this set includes trips which have melbourne central as a stop. for reference, the stop id for MC is 19842.

# In[20]:


#initialising set
mc_trips = set()
for i in range(len(stop_times)):
    #adding trip to mc_trips if it is on a weekday, and contains MC
    if (stop_times.loc[i, "trip_id"] in trip_set) & (str(stop_times.loc[i, "stop_id"])== "19842"):
        mc_trips.add(stop_times.loc[i, "trip_id"])


# In[21]:


print(len(trip_set), len(mc_trips))


# as can be seen, the set of possible trips has shrunk from 23809 to 5963, this will help speed up the processing. <br>
# when running the code it became apparent that some of the timestamps are in the incorrect format. specifically, some of the timestamps dont exist (e.g. 24:12:00). this needs to be rectified before the trip duration is calculated. this is done by converting 24 to 00, where these errors occur.

# In[22]:


n = 0
for i in range(len(stop_times)):
    if int(stop_times.loc[i, "arrival_time"][0:2])==24:
        mins = str(stop_times.loc[i, "arrival_time"][2:])
        n+=1
        stop_times.loc[i, "arrival_time"] = "00" + mins


# In[23]:


print(len(stop_times), n)


# although a lot of alterations are made to the data set, it only comprises 1.86% of the total data. this is acceptable. 

# ### 5.2 - calculation of the shortest trip
# to improve processing speed, the dataframes are converted to numpy arrays. So they can be converted back to dataframes later, the column names are saved to a list

# In[24]:


cd_np = clean_data.to_numpy()
st_np = stop_times.to_numpy()
column_names = clean_data.columns.tolist()


# now the data has been converted to an array, the travel times can be calculated. this is done by looping through the data, isolated the closest stop and finding all trips which go through this stop. each trip is then followed through to completion (using the sequence number) and if the trip reaches MC before it finishes the duration of the trip is addded to a running total. at the point the number of trips recorded is also incremented. once all trips have been found, the times are averaged and saved to a dictionary. if no trips run from that stop to melbourne central, then the travel time is saved as "not available". for reference, the following values correspond to the these indexes:
# - closest_station = 6
# - travel_min_to_MC = 8
# - direct_journey_flag = 9
# - trip_id = 0
# - arrival_time = 1
# - departure_time = 2
# - stop_id = 3
# - stop_sequence = 4

# In[25]:


trip_dict = {}
#if closest station is MC, set time to 0
trip_dict["19842"] = 0
#looping through data
for j in range(len(cd_np)):
    stop_id = str(cd_np[j, 6])
    if (j%100 == 0) and (j>1):
        print("FIRST {} DONE!".format(j))
    if stop_id not in trip_dict.keys():
        #initialising time list
        time_total = 0
        number_trips = 0
        #looping through stops
        for i in range(len(st_np)):
            #if the stop is the nearest, checking that trip runs on a weekday
            if str(st_np[i, 3]) == stop_id:
                hour = int(st_np[i, 2][0:2])
                if (st_np[i, 0] in mc_trips) & (hour > 6) & (hour < 9):
                    #if the train is running at the right time, following route to see if it ends up at MC
                    seq_begin = int(st_np[i, 4])
                    for k in range(1, 32):
                        seq_end = int(st_np[i+k, 4])
                        if seq_end<seq_begin:
                            break
                        elif str(st_np[i+k, 3])=="19842":
                            delta_t = datetime.datetime.strptime(st_np[i+k, 1], '%H:%M:%S') - datetime.datetime.strptime(st_np[i, 2], '%H:%M:%S')
                            time_total += (delta_t.seconds/60)
                            number_trips += 1
                            #print("MC FOUND, time {} added, total now: {}".format(delta_t.seconds/60, time_total))
        if number_trips == 0:
            trip_dict[stop_id] = "not available"
        else:
            trip_dict[stop_id] = round(time_total/number_trips)
        
print("COMPLETE!")


# ### 5.3 - filling in the dataframe
# now dictionary of travel times has been made, the data can be filled in

# In[26]:


for i in range(len(cd_np)):
    stop_id = str(cd_np[i, 6])
    for key, value in trip_dict.items():
        if key == stop_id:
            cd_np[i, 8] = value
            
cd_np[:2, :]


# since the calculations are now complete, the data can be converted back to dataframe (as its my preferred sturcture for handling data)

# In[27]:


#overriding the original dataset (which does not have complete trip duration columns)
clean_data = pd.DataFrame(cd_np, columns = column_names)


# where travel times were calculated, direct journeys exist. the the trip duration is given as "not available" no direct journeys were possible (to MC). hence, then values in the "travel_min_to_MC" column can be used to fill in the "direct_journey_flag". where no data was available (like the closest train station could not be determined), the travel_min will be listed as -1, and the direct_journey_flag should be given this value also, as they are the default values for this column.

# In[28]:


for i in range(len(clean_data)):
    #finding travel_min value
    value = str(clean_data.loc[i, "travel_min_to_MC"])
    if value == "not available":
        clean_data.loc[i, "direct_journey_flag"] = 0
    elif value == "-1":
        clean_data.loc[i, "direct_journey_flag"] = "-1"
    else:
        clean_data.loc[i, "direct_journey_flag"] = 1


# to check the process has been successful, the first few lines of the dataframe and the unique values of the journey flag are checked

# In[29]:


clean_data.loc[:, "direct_journey_flag"].unique()


# In[30]:


clean_data.head()


# the calculations appear to be successful!

# ## Part 6 - web scraping
# now only the covid data remains to be filled in. this data is available at https://covidlive.com.au/ and needs to be scraped to retrieve it. data is available for each LGA, so once scraped, it is stored in a dictionary with the LGA as the key. the data that is required is the case numbers for the 30th september, the running average for the last 14, 30, and 60 days.

# In[31]:


#identifying unique Lgas for which the covid data must be retrieved
lga_values = clean_data.loc[:, "Lga"].unique().tolist()
#initialising dictionary
covid_dict = {}


# the following code (used to scrape the web data) was retrieved from pluralsight and altered to fit this setting (Singhal, 2021). it uses requests to fetch the data, and beautifulsoup to parse. the table can then be extracted, and the headers and rows identified by their tags (th and tr respectively). the headers and row data are used to produce a dictionary for each row of the table. this list of dictionaries can then be easily converted to a dataframe. from this dataframe, the sep 30 data, last 14 days, last 30 days, and last 60 days can all be extracted. these are saved as separate dataframes, and the difference between rows calculated. this corresponds to the daily change in cases. its these values that are averaged and rounded to produce the data for "last_14_days_cases", "last_30_days_cases", "last_60_days_cases", and isolated to give "30_sep_cases". 

# In[32]:


for LGA in lga_values: 
    lga = LGA.lower().replace(" ", "-")
    if not lga == "not available":
        url = "https://covidlive.com.au/vic/" + lga

        #fetching and parsing content
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "lxml")

        #extracting the table which contains case data 
        table = soup.find("table", class_="DAILY-CASES-BY-LGA")

        #getting column headers of table
        t_headers = []
        if table:
            for th in table.find_all("th"):
                #stripping headers
                t_headers.append(th.text.replace('\n', ' ').strip())

            #getting row data
            table_info = []
            for tr in table.find_all("tr"):
                t_row = {}
                #storing datalist of dictionaries (where headers are keys)
                for td, th in zip(tr.find_all("td"), t_headers): 
                    t_row[th] = td.text.replace('\n', '').strip()
                #adding row dictionary to list
                table_info.append(t_row)

            #converting data dictionaries to dataframe
            lga_data = pd.DataFrame(table_info)
            #removing null rows
            lga_data.dropna(subset = ["DATE"], inplace=True)
            #extracting 30th september data
            sep_30_ind = int(lga_data.index[lga_data['DATE'] == "30 Sep"].tolist()[0])
            sep_29_ind = sep_30_ind +1
            sep_30_data = lga_data.loc[sep_30_ind:sep_29_ind, :].apply(lambda x: x.str.replace(',',''))
            sep_30_data = sep_30_data.reset_index(drop=True)
            #creating dataframe for 14 day period
            first_14_ind = sep_30_ind + 15
            last_14_data = lga_data.loc[sep_29_ind:first_14_ind, :].apply(lambda x: x.str.replace(',',''))

            #creating dataframe for 30 day period
            first_30_ind = sep_30_ind + 31
            last_30_data = lga_data.loc[sep_29_ind:first_30_ind, :].apply(lambda x: x.str.replace(',',''))

            #creating dataframe for 60 day period
            first_60_ind = sep_30_ind + 61
            last_60_data = lga_data.loc[sep_29_ind:first_60_ind, :].apply(lambda x: x.str.replace(',',''))

            #converting formats
            sep_30_data.loc[:,"CASES"] = pd.to_numeric(sep_30_data.loc[:,"CASES"])
            last_14_data.loc[:,"CASES"] = pd.to_numeric(last_14_data.loc[:,"CASES"])
            last_30_data.loc[:,"CASES"] = pd.to_numeric(last_30_data.loc[:,"CASES"])
            last_60_data.loc[:,"CASES"] = pd.to_numeric(last_60_data.loc[:,"CASES"])
            #calculating change in cases
            sep_30_data.loc[:,"daily cases"] = sep_30_data.loc[:,"CASES"].diff(periods=-1)
            last_14_data.loc[:,"daily cases"] = last_14_data.loc[:,"CASES"].diff(periods=-1)
            last_30_data.loc[:,"daily cases"] = last_30_data.loc[:,"CASES"].diff(periods=-1)
            last_60_data.loc[:,"daily cases"] = last_60_data.loc[:,"CASES"].diff(periods=-1)
            #calculating rounded averages
            sep_30 = round(sep_30_data.loc[0,"daily cases"])
            avg_14 = round(last_14_data.loc[:,"daily cases"].mean())
            avg_30 = round(last_30_data.loc[:,"daily cases"].mean())
            avg_60 = round(last_60_data.loc[:,"daily cases"].mean())
            covid_dict[lga] = (sep_30, avg_14, avg_30, avg_60)


# In[33]:


for i in range(len(clean_data)):
    #converting lga to appropriate format for comparison
    lga = clean_data.loc[i, "Lga"].lower().replace(" ", "-")
    for key, value in covid_dict.items():
        #filling in data based on dictionary values
        if key == lga:
            clean_data.loc[i, "30_sep_cases"] = value[0]
            clean_data.loc[i, "last_14_days_cases"] = value[1]
            clean_data.loc[i, "last_30_days_cases"] = value[2]
            clean_data.loc[i, "last_60_days_cases"] = value[3]
#printing data for inspection
clean_data.head()


# as can be seen, all the required data has been filled in, so task 1 is complete!

# ## Part 7 - fixing data to match sample output
# although the data has been filled in, not all the formatting matches the sample output. this needs to be rectified before the data is written to a csv

# In[34]:


#renaming columns
clean_data = clean_data.rename(columns={'Lga': 'lga'})

#making property_i index column
clean_data.set_index('property_id', inplace=True)

#rounding latitude and longitude to 6 decimal places
clean_data["lat"] = clean_data["lat"].apply(lambda x: round(x, 6))
clean_data["lng"] = clean_data["lng"].apply(lambda x: round(x, 6))


# ## Part 8 - writing data to csv
# now the data formatting has been fixed, it can be exported to a csv

# In[35]:


clean_data.to_csv('cleaned_data.csv')


# # Task 2: data reshaping
# task 2 involves the exploration of the data with the goal of creating a regression model for predicting the sep 30th cases. the predictors in this case are the rounded average cases for the preceeding 14, 30, and 60 days. in order to build a regression model, the data first needs to be inspected to determine if any scaling or transofrmations are necessary. this is achieved using some of the code made available during tutorials (Haqqani, 2021).

# In[36]:


plotting = clean_data[clean_data.last_30_days_cases != "not available"]
plotting = plotting[plotting.last_60_days_cases != "not available"]
plotting = plotting[plotting.last_14_days_cases != "not available"]
plotting = plotting[plotting["30_sep_cases"] != "not available"]


# In[37]:


#exploratory plotting
f = plt.figure(figsize=(8,6))

plt.scatter(plotting["last_14_days_cases"], plotting["30_sep_cases"],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(plotting["last_30_days_cases"], plotting["30_sep_cases"], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(plotting["last_60_days_cases"], plotting["30_sep_cases"],
        color='blue', label='60 day average', alpha=0.3)

plt.title('average of covid cases as predictor for sept 30th')
plt.xlabel('avg number of cases over last days')
plt.ylabel('30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# as can be seen, scaling is required. this makes sense as the larger the period which is averaged, the more stable and  lower variance the data should be. as the data requires scaling, model training should only be performed on the appropriately scaled and transformed data. to determine the best method for scaling, multiple techniques can be tested. these include minmax scaling, and z-score normalisaton.

# In[38]:


#creating arrays of scaled data
#z-score normalisation
std_scale = preprocessing.StandardScaler().fit(plotting[['last_14_days_cases','last_30_days_cases','last_60_days_cases', '30_sep_cases']])
df_std = std_scale.transform(plotting[['last_14_days_cases','last_30_days_cases','last_60_days_cases', '30_sep_cases']])
#minmax scaling
minmax_scale = preprocessing.MinMaxScaler().fit(plotting[['last_14_days_cases','last_30_days_cases','last_60_days_cases', '30_sep_cases']])
df_minmax = minmax_scale.transform(plotting[['last_14_days_cases','last_30_days_cases','last_60_days_cases', '30_sep_cases']])


# In[39]:


#Plotting the z-score normalised data values

get_ipython().run_line_magic('matplotlib', 'inline')

f = plt.figure(figsize=(8,6))

plt.scatter(df_std[:,0], df_std[:,3],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(df_std[:,1], df_std[:,3], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(df_std[:,2], df_std[:,3],
        color='blue', label='60 day average', alpha=0.3)

plt.title('Standardized covid cases as predictor for sept 30th')
plt.xlabel('avg number of cases')
plt.ylabel('30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# In[40]:


#Plotting the minmax scaled data values

get_ipython().run_line_magic('matplotlib', 'inline')

f = plt.figure(figsize=(8,6))

plt.scatter(df_minmax[:,0], df_minmax[:,3],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(df_minmax[:,1], df_minmax[:,3], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(df_minmax[:,2], df_minmax[:,3],
        color='blue', label='60 day average', alpha=0.3)

plt.title('min-max covid cases as predictor for sept 30th')
plt.xlabel('avg number of cases')
plt.ylabel('30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# For the purposes of creating a linear regression model, it will be more useful to retain the distribution of the data. for this reason, minmax scaling is a better alternative to z-score normalisation (Singh et al., 2021). studies also suggest that minmax performs better than other scaling methods in most scenarios, and that minmax has the best computational time complexity.

# now the data has been scaled, transformations need to be applied to improve the linearity of the data. square root, square, log, and boxcox are all possible, suitable transformations. however, boxcox requires strictly positive values (which makes it a bias transformation in this setting) 

# In[41]:


#saving minmax data to dataframe 
plotting = pd.DataFrame(df_minmax, columns = ['last_14_days_cases','last_30_days_cases','last_60_days_cases', '30_sep_cases'])


# In[42]:


#testing sqrt transformation
plotting['rl14']=plotting["last_14_days_cases"]**(1/2)
plotting['rl30']=plotting["last_30_days_cases"]**(1/2)
plotting['rl60']=plotting["last_60_days_cases"]**(1/2)
plotting['r30']=plotting["30_sep_cases"]**(1/2)

f = plt.figure(figsize=(8,6))

plt.scatter(plotting['rl14'], plotting['r30'],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(plotting['rl30'], plotting['r30'], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(plotting['rl60'], plotting['r30'],
        color='blue', label='60 day average', alpha=0.3)

plt.title('sqrt transformation of cases as predictor for sept 30th')
plt.xlabel('sqrt of avg number of cases')
plt.ylabel('sqrt of 30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# In[43]:


#testing square transformation
plotting['sql14']=plotting["last_14_days_cases"]**(2)
plotting['sql30']=plotting["last_30_days_cases"]**(2)
plotting['sql60']=plotting["last_60_days_cases"]**(2)
plotting['sq30']=plotting["30_sep_cases"]**(2)

f = plt.figure(figsize=(8,6))

plt.scatter(plotting['sql14'], plotting['sq30'],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(plotting['sql30'], plotting['sq30'], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(plotting['sql60'], plotting['sq30'],
        color='blue', label='60 day average', alpha=0.3)

plt.title('square transformation of cases as predictor for sept 30th')
plt.xlabel('square of avg number of cases')
plt.ylabel('square of 30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# In[44]:


#testing log transformation
plotting = plotting[["last_14_days_cases","last_30_days_cases","last_60_days_cases","30_sep_cases"]].apply(pd.to_numeric)
plotting['ll14']=np.log(plotting["last_14_days_cases"])
plotting['ll30']=np.log(plotting["last_30_days_cases"])
plotting['ll60']=np.log(plotting["last_60_days_cases"])
plotting['l30']=np.log(plotting["30_sep_cases"])

f = plt.figure(figsize=(8,6))

plt.scatter(plotting['ll14'], plotting['l30'],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(plotting['ll30'], plotting['l30'], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(plotting['ll60'], plotting['l30'],
        color='blue', label='60 day average', alpha=0.3)

plt.title('log transformation of cases as predictor for sept 30th')
plt.xlabel('log of avg number of cases')
plt.ylabel('log of 30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# In[45]:


#testing boxcox transformation
plotting["last_14_days_cases"] = plotting["last_14_days_cases"].apply(abs)
plotting["last_30_days_cases"] = plotting["last_30_days_cases"].apply(abs)
plotting["last_60_days_cases"] = plotting["last_60_days_cases"].apply(abs)
plotting["30_sep_cases"] = plotting["30_sep_cases"].apply(abs)
plotting = plotting[plotting.last_14_days_cases != 0]
plotting = plotting[plotting.last_30_days_cases != 0]
plotting = plotting[plotting.last_60_days_cases != 0]
plotting = plotting[plotting["30_sep_cases"] != 0]

plotting['bxl14'], _ = stats.boxcox(plotting["last_14_days_cases"])
plotting['bxl30'], _ = stats.boxcox(plotting["last_30_days_cases"])
plotting['bxl60'], _ = stats.boxcox(plotting["last_60_days_cases"])
plotting['bx30'], _ = stats.boxcox(plotting["30_sep_cases"])

f = plt.figure(figsize=(8,6))

plt.scatter(plotting['bxl14'], plotting['bx30'],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(plotting['bxl30'], plotting['bx30'], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(plotting['bxl60'], plotting['bx30'],
        color='blue', label='60 day average', alpha=0.3)

plt.title('boxcox transformation of cases as predictor for sept 30th')
plt.xlabel('boxcox of avg number of cases')
plt.ylabel('boxcox of 30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# as can be seen from the plots, boxcox and log provide good ways to scale the data. However, boxcox requires strictly positive data (Box & Cox, 1964). this biases the data, as the change in covid cases can be negative, or 0. hence, for this dataset boxcox is unsuitable. the square root transformation also appears promisingly linear. however, the data is unevenly distributed. to determine which method (square root or log) is better. two linear models can be constructed, and the r squared and RMSE scores compared to determine the best transformation.

# training model on log data

# In[46]:


#splitting data into independent and dependent variables
lx = pd.DataFrame(plotting,columns=["ll14","ll30","ll60"]) 
ly = pd.DataFrame(plotting,columns=["l30"])
#splitting into test and training sets:
x_train, x_test, y_train, y_test = train_test_split(lx, ly, test_size = 0.3, random_state = 100)
# creating a Linear regression object and training it on the data
log_model = LinearRegression().fit(x_train,y_train) # (Pedregosa et al., 2011)
#predicting outcomes on the test set and using them to evaluate the model
y_prediction =  log_model.predict(x_test)
predictions = y_test.copy()
predictions['prediction'] = y_prediction
predictions.head()
#printing accuracy metrics
print("r2 score is: ", r2_score(y_test,y_prediction))
print("root_mean_squared error is: ",np.sqrt(mean_squared_error(y_test,y_prediction)))


# training model on square root data. <br>
# note: although the root variables have already been saved to the dataframe, they disappear at this point. Hence, they need to be recalculated before the model can be trained on them

# In[47]:


plotting['rl14']=plotting["last_14_days_cases"]**(1/2)
plotting['rl30']=plotting["last_30_days_cases"]**(1/2)
plotting['rl60']=plotting["last_60_days_cases"]**(1/2)
plotting['r30']=plotting["30_sep_cases"]**(1/2)

#splitting data into independent and dependent variables
rx = pd.DataFrame(plotting,columns=["rl14","rl30","rl60"]) 
ry = pd.DataFrame(plotting,columns=["r30"])
#splitting into test and training sets:
x_train, x_test, y_train, y_test = train_test_split(rx, ry, test_size = 0.3, random_state = 100)
# creating a Linear regression object and training it on the data
sqrt_model = LinearRegression().fit(x_train,y_train) # (Pedregosa et al., 2011)
#predicting outcomes on the test set and using them to evaluate the model
y_prediction =  sqrt_model.predict(x_test)
predictions = y_test.copy()
predictions['prediction'] = y_prediction
predictions.head()
#printing accuracy metrics
print("r2 score is: ", r2_score(y_test,y_prediction))
print("root_mean_squared error is: ",np.sqrt(mean_squared_error(y_test,y_prediction)))


# as can be seen by the r squared and RMSE the square root model is better, having both a higher r squared and a much lower RMSE. to determine if square root is best method of transforming the data other powers need to be tested (like cube root). this is done for fractions between 1 and 1/10.

# In[48]:


#initialising dictionary
score_dict = {}
for i in range(1,11):
    for j in range(1,i+1):
        fraction = j/i
        test_data = pd.DataFrame()
        test_data['l14']=plotting["last_14_days_cases"]**(fraction)
        test_data['l30']=plotting["last_30_days_cases"]**(fraction)
        test_data['l60']=plotting["last_60_days_cases"]**(fraction)
        test_data['30']=plotting["30_sep_cases"]**(fraction)
        #splitting data into independent and dependent variables
        x = pd.DataFrame(test_data,columns=["l14","l30","l60"]) 
        y = pd.DataFrame(test_data,columns=["30"])
        #splitting into test and training sets:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 100)
        # creating a Linear regression object and training it on the data
        model = LinearRegression().fit(x_train,y_train) # (Pedregosa et al., 2011)
        #predicting outcomes on the test set and using them to evaluate the model
        y_prediction = model.predict(x_test)
        predictions = y_test.copy()
        predictions['prediction'] = y_prediction
        #saving accuracy metrics
        score_dict[fraction] = [round(r2_score(y_test,y_prediction),5),round(np.sqrt(mean_squared_error(y_test,y_prediction)),5)]
        


# In[49]:


print(score_dict)


# finding max r squared

# In[50]:


print(max(score_dict, key=score_dict.get))


# finding min RMSE

# In[51]:


print(min(score_dict, key=score_dict.get))


# as can be seen by the dictionary outputs, the transformation which reduces RMSE is not the same as the function which maximises r squared. more explicitly, the transformation which minimises RMSE is the tenth power (^1/10), but that which maximises r squared is no transformation (just scaling). hence, the importance of these two metrics needs to be weighted to determine the best model. For interest, the two plots are compared below, showing the linear line of best fit.

# In[52]:


#testing sqrt transformation
plotting['minl14']=plotting["last_14_days_cases"]**(1/10)
plotting['minl30']=plotting["last_30_days_cases"]**(1/10)
plotting['minl60']=plotting["last_60_days_cases"]**(1/10)
plotting['min30']=plotting["30_sep_cases"]**(1/10)

f = plt.figure(figsize=(8,6))

plt.scatter(plotting['minl14'], plotting['min30'],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(plotting['minl30'], plotting['min30'], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(plotting['minl60'], plotting['min30'],
        color='blue', label='60 day average', alpha=0.3)
plotting['min_mean'] = plotting.loc[:,["minl14","minl30","minl60"]].mean(axis = 1)
plt.plot(np.unique(plotting['min_mean']), np.poly1d(np.polyfit(plotting['min_mean'], plotting['min30'], 1))(np.unique(plotting['min_mean'])))
plt.title('^1/10 transformation of cases as predictor for sept 30th')
plt.xlabel('^1/10 of avg number of cases')
plt.ylabel('^1/10 of 30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# In[53]:


#testing sqrt transformation
plotting['l14']=plotting["last_14_days_cases"]
plotting['l30']=plotting["last_30_days_cases"]
plotting['l60']=plotting["last_60_days_cases"]
plotting['30']=plotting["30_sep_cases"]

f = plt.figure(figsize=(8,6))

plt.scatter(plotting['l14'], plotting['30'],
        color='green', label='14 day average', alpha=0.5)

plt.scatter(plotting['l30'], plotting['30'], color='red',
         label='30 day average', alpha=0.3)

plt.scatter(plotting['l60'], plotting['30'],
        color='blue', label='60 day average', alpha=0.3)

plotting['mean'] = plotting.loc[:,["l14","l30","l60"]].mean(axis = 1)
plt.plot(np.unique(plotting['mean']), np.poly1d(np.polyfit(plotting['mean'], plotting['30'], 1))(np.unique(plotting['mean'])))

plt.title('cases as predictor for sept 30th')
plt.xlabel('avg number of cases')
plt.ylabel('30th sept number cases')
plt.legend(loc='upper left')
plt.grid()
plt.tight_layout()

plt.show()


# due to the comparative distributions of the data^(1/10) and the raw, scaled data, the 10th root appears more reliable. as the data is more evenly spread. this means the line of best fit is less reliant on the outlier point. thus the tenth root data is what is used when training the final model here. however, the suitability of one model over the other depends on the application of the model. all of the powers (between 1 and 1/10) proved to be good transformations of the data (as shown by the high r-squared and low RMSE values in the score_dict). hence, the final model chosen should be picked based on the application, rather than purely off the RMSE or r-squared scores. Grace-Martin (2021) indicates the r-squared can be more relevant when making predictions, but RMSE is more useful for proving relationships. this should be kept in mind when choosing the best model for the application.

# to improve the accuracy of the model (in its application on unseen data), it is retrained on the full dataset. this avoids any patterns in the full dtaaset being missed

# In[54]:


#retraining model on full dataset
x = pd.DataFrame(test_data,columns=["minl14","minl30","minl60"]) 
y = pd.DataFrame(test_data,columns=["min30"])
#creating a model
final_model = LinearRegression()
#training it on the data
final_model.fit(rx,ry) # (Pedregosa et al., 2011)


# this is the final model produced, and could be used to predict the september 30th case data based off the preceeding 14, 30, and 60 day averages.

# # References
# Pedregosa, F., Varoquaux, G.,  Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J.,  Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E. (2011), “Scikit-learn: Machine Learning in Python”, Journal of Machine Learning Research 12, pp. 2825-2830
# <br>
# Shopeju, H. (2019). How to Parse XML Files Using Python’s BeautifulSoup. Retrieved 1 November 2021, from https://linuxhint.com/parse_xml_python_beautifulsoup/ 
# <br>
# Preste, R. (2019). From XML to Pandas dataframes. Retrieved 1 November 2021, from https://medium.com/@robertopreste/from-xml-to-pandas-dataframes-9292980b1c1c
# <br>
# Harris, C. (2013). Check if a geopoint with latitude and longitude is within a shapefile. Retrieved 1 November 2021, from https://stackoverflow.com/questions/7861196/check-if-a-geopoint-with-latitude-and-longitude-is-within-a-shapefile
# <br>
# Phaseit Inc. (2021). PyPDF2 1.26.0 documentation. Retrieved 1 November 2021, from https://pythonhosted.org/PyPDF2/About%20PyPDF2.html
# <br>
# Dib, F. (2021). regex101: build, test, and debug regex. Retrieved 1 November 2021, from https://regex101.com/
# <br>
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
# <br>
# Singhal, G. (2021). Python BeautifulSoup Web Scraping. Retrieved 1 November 2021, from https://www.pluralsight.com/guides/extracting-data-html-beautifulsoup
# <br>
# Singh, Abhilash & Gaurav, Kumar & Rai, Atul & Beg, Zafar. (2021). Machine Learning to Estimate Surface Roughness from Satellite Images. Remote Sensing. 13. 1-27. 10.3390/rs13193794
# <br>
# Box, G. E. P. and Cox, D. R. (1964). An analysis of transformations, Journal of the Royal Statistical Society, Series B, 26, 211-252.
# <br>
# Haqqani, M. (2021). Tute 11 - Data Normalisation and Transformation. Tutorial, Monash University.
# <br>
# Grace-Martin, K. (2021). Assessing the Fit of Regression Models. Retrieved 1 November 2021, from https://www.theanalysisfactor.com/assessing-the-fit-of-regression-models/

# In[ ]:




