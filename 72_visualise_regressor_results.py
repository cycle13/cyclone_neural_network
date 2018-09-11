import cv2
import pickle

image = pickle.load(open('demo/cyclone_test_result_x.p', 'rb'))
actual = pickle.load(open('demo/cyclone_test_result_y.p', 'rb'))
label = pickle.load(open('demo/cyclone_test_result_p.p', 'rb'))

print(image.shape)

def colour(i, s, c):
    i[s[0], s[1]] = c
    i[s[0]-1, s[1]] = c
    i[s[0]+1, s[1]] = c
    i[s[0], s[1]-1] = c
    i[s[0], s[1]+1] = c
    i[s[0]-1, s[1]-1] = c
    i[s[0]+1, s[1]+1] = c
    i[s[0]-1, s[1]+1] = c
    i[s[0]-1, s[1]-1] = c


i = 0
while i < image.shape[0]:
    img = image[i]
    l = label[i].astype(int)
    a = actual[i].astype(int)



    print(img.shape)
    colour(img,[a[0], a[1]], [255,0,0])
    colour(img,[l[0], l[1]], [255, 0, 255])


    cv2.imwrite('demo/color_img{}.jpg'.format(i), img)

    i+=1

