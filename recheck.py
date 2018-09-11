import os 
import re 

imgDir = 'images1'

files = os.listdir(imgDir)

def coord_transform(data):
    r = re.compile("([0-9]+)([a-zA-Z]+)")
    m = r.match(data)

    numeric = int(m.group(1))
    direction = m.group(2)
    
    # Check if numeric is correct
    if numeric > 180:
        print("Invalid number range: " + data)
        return False
    
    if direction not in ['W','E','N','S']:
        print("Invalid direction: " + data)
        return False
    
    # Convert to pre-defined numbering system for coordinates
    if numeric == 0:
        return 0
    
    if direction in ['N','E']:
        return numeric

    if direction in ['W','S']:
        return -numeric

dif_lat = []
dif_lon = []
location = []
i = 0
while i < len(files):
    if files[i].endswith('labels.tsv'):
        with open(imgDir + '/' +  files[i],'r') as f:
            labels = f.readlines()[0].strip()
        labels = labels[1:-1].split(',')
        top = labels[0].split("'")[1]
        bottom = labels[1].split("'")[1]
        left = labels[2].split("'")[1]
        right = labels[3].split("'")[1]
        
        bottom_t = coord_transform(bottom)
        top_t = coord_transform(top)
        left_t = coord_transform(left)
        right_t = coord_transform(right)

        dif_lat.append(top_t - bottom_t)
        lon_dif = right_t - left_t
        if lon_dif < 0:
            print(files[i],right_t, left_t,lon_dif)
            print(right_t+360-left_t)
        dif_lon.append(lon_dif)
        location.append(i)
    i+=1
