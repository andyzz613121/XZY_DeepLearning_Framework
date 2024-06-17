import cv2
# for num in [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]:
#     name1 = 'label' + str(num) + '_gray.tif'
#     name2 = 'label' + str(num) + '_edge.tif'
#     img = cv2.imread(name1)
#     img = cv2.Canny(img,0,10)
#     img = img/255
#     img = img.astype(int)
#     cv2.imwrite(name2,img)

for num in [1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]:
    name1 = 'label_edge\\label' + str(num) + '_edge.tif'
    name2 = 'label_edge_dilate\\label' + str(num) + '_edge.tif'
    img = cv2.imread(name1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(img,kernel)
    img = img.astype(int)
    cv2.imwrite(name2,img)

def label2edge(img_path, out_path):
    img = cv2.imread(img_path)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.dilate(img, kernel)
    img = img.astype(int)
    cv2.imwrite(out_path, img)
