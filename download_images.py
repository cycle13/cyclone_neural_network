import urllib.request
import time

def gen_file_name(link):
    # Filename components
    fc = link.split('/')
    year = fc[4][2:]
    basin = fc[5]
    cyclone_name = fc[6].split('.')[1]
    date = fc[10].split('.')[0]
    t = fc[10].split('.')[1]
    filename = year + '_' + basin + '_' + cyclone_name + '_' + date + '_' + t + '.jpg'
    return filename

def main():
    with open('need_download.txt', 'r') as f:
        links = f.readlines()
    
    i = 0
    total_number_of_links = len(links)
    while i < total_number_of_links:
        filename = gen_file_name(links[i])
        done = False

        while not done:
            try:
                urllib.request.urlretrieve(links[i],'sequence_images/' +  filename)
                done = True
            except:
                print('Waiting for retry')
                time.sleep(30)

        i+=1
        
        print('{}/{}:{}'.format(i,total_number_of_links,filename))

if __name__ == '__main__':
    main()