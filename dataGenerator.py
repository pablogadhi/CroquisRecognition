import os
from PIL import Image
import numpy as np

os.chdir('data/MickeyMouse')
mkms = []
for f in os.listdir('.'):
    img = Image.open(f).convert('L')
    mkms.append(np.array(img))

mkms = np.array(mkms).reshape(1975, 784)
np.random.shuffle(mkms)

os.chdir('../Egg')
eggs = []
for f in os.listdir('.'):
    img = Image.open(f).convert('L')
    img_arr = np.array(img)
    if img_arr.shape == (28, 28):
        eggs.append(img_arr)

x_dim = len(eggs)
eggs = np.array(eggs).reshape(x_dim, 784)
np.random.shuffle(eggs)
eggs = eggs[:1975, :]
print(eggs.shape)

os.chdir('../QuestionMark')
qms = []
for f in os.listdir('.'):
    try:
        img = Image.open(f).convert('L')
    except:
        pass
    img_arr = np.array(img)
    if img_arr.shape == (28, 28):
        qms.append(img_arr)

x_dim = len(qms)
qms = np.array(qms).reshape(x_dim, 784)
np.random.shuffle(qms)
qms = qms[:1975, :]
print(qms.shape)

os.chdir('../SadFace')
sfs = []
for f in os.listdir('.'):
    img = Image.open(f).convert('L')
    img_arr = np.array(img)
    if img_arr.shape == (28, 28):
        sfs.append(img_arr)

x_dim = len(sfs)
sfs = np.array(sfs).reshape(x_dim, 784)
np.random.shuffle(sfs)
sfs = sfs[:1975, :]
print(sfs.shape)

os.chdir('..')

circles = np.load('full_numpy_bitmap_circle.npy', allow_pickle=True)
np.random.shuffle(circles)
circles = circles[:1975, :]

houses = np.load('full_numpy_bitmap_house.npy', allow_pickle=True)
np.random.shuffle(houses)
houses = houses[:1975, :]

smileys = np.load('full_numpy_bitmap_smiley_face.npy', allow_pickle=True)
np.random.shuffle(smileys)
smileys = smileys[:1975, :]

squares = np.load('full_numpy_bitmap_square.npy', allow_pickle=True)
np.random.shuffle(squares)
squares = squares[:1975, :]

trees = np.load('full_numpy_bitmap_tree.npy', allow_pickle=True)
np.random.shuffle(trees)
trees = trees[:1975, :]

triangles = np.load('full_numpy_bitmap_triangle.npy', allow_pickle=True)
np.random.shuffle(triangles)
triangles = triangles[:1975, :]

train_data = np.vstack((
    circles[:1383, :],
    squares[:1383, :],
    triangles[:1383, :],
    eggs[:1383, :],
    trees[:1383, :],
    houses[:1383, :],
    smileys[:1383, :],
    sfs[:1383, :],
    qms[:1383, :],
    mkms[:1383, :]
))

train_labels = np.vstack((
    np.full((1383, 1), 0),
    np.full((1383, 1), 1),
    np.full((1383, 1), 2),
    np.full((1383, 1), 3),
    np.full((1383, 1), 4),
    np.full((1383, 1), 5),
    np.full((1383, 1), 6),
    np.full((1383, 1), 7),
    np.full((1383, 1), 8),
    np.full((1383, 1), 9),
))

train = np.hstack((train_data, train_labels))
np.random.shuffle(train)
train = np.hsplit(train, [784, 785])
train_data = train[0]
train_labels = train[1]
np.save('train_data.npy', train_data, allow_pickle=True)
np.save('train_labels.npy', train_labels, allow_pickle=True)

cv_data = np.vstack((
    circles[1383:1679, :],
    squares[1383:1679, :],
    triangles[1383:1679, :],
    eggs[1383:1679, :],
    trees[1383:1679, :],
    houses[1383:1679, :],
    smileys[1383:1679, :],
    sfs[1383:1679, :],
    qms[1383:1679, :],
    mkms[1383:1679, :]
))

cv_labels = np.vstack((
    np.full((296, 1), 0),
    np.full((296, 1), 1),
    np.full((296, 1), 2),
    np.full((296, 1), 3),
    np.full((296, 1), 4),
    np.full((296, 1), 5),
    np.full((296, 1), 6),
    np.full((296, 1), 7),
    np.full((296, 1), 8),
    np.full((296, 1), 9),
))

cv = np.hstack((cv_data, cv_labels))
np.random.shuffle(cv)
cv = np.hsplit(cv, [784, 785])
cv_data = cv[0]
cv_labels = cv[1]
np.save('cv_data.npy', cv_data, allow_pickle=True)
np.save('cv_labels.npy', cv_labels, allow_pickle=True)

test_data = np.vstack((
    circles[1679:1975, :],
    squares[1679:1975, :],
    triangles[1679:1975, :],
    eggs[1679:1975, :],
    trees[1679:1975, :],
    houses[1679:1975, :],
    smileys[1679:1975, :],
    sfs[1679:1975, :],
    qms[1679:1975, :],
    mkms[1679:1975, :]
))

test_labels = np.vstack((
    np.full((296, 1), 0),
    np.full((296, 1), 1),
    np.full((296, 1), 2),
    np.full((296, 1), 3),
    np.full((296, 1), 4),
    np.full((296, 1), 5),
    np.full((296, 1), 6),
    np.full((296, 1), 7),
    np.full((296, 1), 8),
    np.full((296, 1), 9),
))

test = np.hstack((test_data, test_labels))
np.random.shuffle(test)
test = np.hsplit(test, [784, 785])
test_data = test[0]
test_labels = test[1]
np.save('test_data.npy', test_data, allow_pickle=True)
np.save('test_labels.npy', test_labels, allow_pickle=True)
