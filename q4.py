# %load q4.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# question 4, part 1
def svd_compress(image_name, k):
    img = mpimg.imread(image_name)
    gray = rgb2gray(img)

    U, S, Vt = np.linalg.svd(gray)
    # your code goes here
    if k == 10:
        np.set_printoptions(threshold=np.nan)
        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)
        print(np.diag(S[:k]))
        print("shape of S: "+str(S.shape)+"shape of Vt: "+str(Vt.shape)+"shape of U: "+str(U.shape))
    recons_img = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(Vt[:k, :])

    fig = plt.figure(figsize=(7, 5))
    plt.imshow(recons_img, cmap=plt.get_cmap('gray'))
    plt.axis('off')
    plt.title("compressed image by k= " + str(k))
    plt.show()

    return recons_img


# question 4, part 2
# your code goes here
nval = [10, 20, 50, 100, 200]
ipath = "neverland.jpg"
ax = []
row = 1

for k in nval:
    img = svd_compress(ipath, k)

plt.show()

# question 4, part 3
def compute_error(image_name, k):
    img = mpimg.imread(image_name)
    gray = rgb2gray(img)

    U, S, Vt = np.linalg.svd(gray)

    # your code goes here
    recons_img = np.matrix(U[:, :k]) * np.diag(S[:k]) * np.matrix(Vt[:k, :])
    dist = np.linalg.norm(gray - recons_img)
    return dist


# question 4, part 4
# your code goes here
fig = plt.figure(figsize=(6, 5))
y = []
x = []
ax = []
for k in nval:
    y.append(compute_error(ipath, k))
    x.append(k)

ax.append(fig.add_subplot(1, 1, 1))
ax[-1].set_title('reconstrauction error')  # set title
plt.plot(x, y, marker='x')
plt.legend(('recons error'))
plt.title('SVD error')
plt.ylabel('Error')
plt.xlabel('k value')
plt.grid(True)
plt.show()