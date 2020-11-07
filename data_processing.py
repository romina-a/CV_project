import os, cv2
import

path = 'data'

labels_obj = {
    'with_mask': 0,
    'without_mask': 1
}

categories = ['with_mask', 'without_mask']

data = list()
target = list()

for ctg in categories:
    folder = os.path.join(path, ctg)
    imgs = os.listdir(folder)

    for i in imgs:
        i_path = os.path.join(folder, i)
        img = cv2.imread(i_path)

        try:
            resize = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(100,100))
            data.append(resize)
            target.append(labels_obj[ctg])
        except Exception as e:
            print(e)


cv2.imshow("test", data[0])
cv2.waitKey(0)  