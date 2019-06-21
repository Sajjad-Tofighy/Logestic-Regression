import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal



def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# question 3, part 1
def conv2d(image_name, mask):
    img = mpimg.imread(image_name)
    gray = rgb2gray(img)

    # show gray scale image, only for test
    # plt.imshow(gray, cmap = plt.get_cmap('gray'))
    # plt.show()

    # your code goes here
    kernel = np.array(mask)

    output = np.zeros_like(gray)  # convolution output

    for x in range(gray.shape[1] - 3):  # Loop over every pixel of the image
        for y in range(gray.shape[0] - 3):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * gray[y:y + 3, x:x + 3]).sum()
    return output


def scipy_conv2d(image_name, mask):
    img = mpimg.imread(image_name)
    gray = rgb2gray(img)

    # your code goes here
    image_sharpen = scipy.signal.convolve2d(gray, np.array(mask), 'same')
    return image_sharpen


# question 3, part 2
# call conv2d for given masks and images
# your code goes here
m1 = [[-1, 0, +1], [-2, 0, 2], [-1, 0, 1]]
m2 = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
ax = []
imgname = ['tesla-cat.jpg', 'tiger.jpg']
row = 1

fig = plt.figure(figsize=(9, 15))
for imgpath in imgname:
    image_sharpen = conv2d(imgpath, m1)
    ax.append(fig.add_subplot(4, 1, row))
    ax[-1].set_title(imgpath + " image: Vertical edge detection")  # set title
    plt.imshow(image_sharpen, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    row += 1
for imgpath in imgname:
    image_sharpen2 = conv2d(imgpath, m2)
    ax.append(fig.add_subplot(4, 1, row))
    ax[-1].set_title(imgpath + " image: Horizontal edge detection")  # set title
    plt.imshow(image_sharpen2, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    row += 1

plt.show()
