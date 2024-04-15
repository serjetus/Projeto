tempPerson = persons.pop(rmv)
tempPerson.extract_caracteristcs()
if len(personsT == 0):
    personsT.append(tempPerson)
else:
    for p in range(len(personsT)):
        if tempPerson.caracterics in personsT[p].caracteristics:
            flagClothes = True

''' splitimages = [self.image]
 for _ in range(2):
     superior, inferior = split_image(splitimages.pop(0))
     splitimages.extend([superior, inferior])
 _, inferior_final = split_image(splitimages.pop(0))
 splitimages.append(inferior_final)

     for i in range(len(splitimages)):
     cv2.imshow("recorte", splitimages[i])
     cv2.waitKey(0)'''




[ultralytics.engine.results.Results object with attributes:

boxes: ultralytics.engine.results.Boxes object
keypoints: ultralytics.engine.results.Keypoints object
masks: None
names: {0: 'person'}
orig_img: array([[[134, 139, 142],
        [135, 140, 143],
        [135, 140, 143],
        ...,
        [136, 142, 143],
        [137, 143, 144],
        [137, 143, 144]],

       [[135, 140, 143],
        [134, 139, 142],
        [136, 141, 144],
        ...,
        [136, 142, 143],
        [137, 143, 144],
        [139, 145, 146]],

       [[139, 144, 147],
        [137, 142, 145],
        [136, 141, 144],
        ...,
        [137, 143, 144],
        [136, 142, 143],
        [135, 139, 142]],

       ...,

       [[127, 128, 135],
        [116, 117, 124],
        [113, 114, 121],
        ...,
        [132, 145, 125],
        [202, 206, 192],
        [207, 209, 197]],

       [[115, 116, 123],
        [117, 118, 125],
        [117, 118, 125],
        ...,
        [157, 155, 156],
        [147, 150, 137],
        [208, 210, 198]],

       [[114, 115, 122],
        [118, 119, 126],
        [118, 119, 126],
        ...,
        [186, 185, 183],
        [152, 152, 146],
        [146, 146, 138]]], dtype=uint8)
orig_shape: (132, 26)
path: 'image0.jpg'
probs: None
save_dir: 'runs\\pose\\predict'
speed: {'preprocess': 2.9931068420410156, 'inference': 434.8573684692383, 'postprocess': 0.9977817535400391}]


A


class for storing and manipulating inference results.


Args:
orig_img(numpy.ndarray): The
original
image as a
numpy
array.
path(str): The
path
to
the
image
file.
names(dict): A
dictionary
of


class names.


boxes(torch.tensor, optional): A
2
D
tensor
of
bounding
box
coordinates
for each detection.
    masks(torch.tensor, optional): A
    3
    D
    tensor
    of
    detection
    masks, where
    each
    mask is a
    binary
    image.
probs(torch.tensor, optional): A
1
D
tensor
of
probabilities
of
each


class for classification task.


keypoints(List[List[float]], optional): A
list
of
detected
keypoints
for each object.

Attributes:
orig_img(numpy.ndarray): The
original
image as a
numpy
array.
orig_shape(tuple): The
original
image
shape in (height, width)
format.
boxes(Boxes, optional): A
Boxes
object
containing
the
detection
bounding
boxes.
masks(Masks, optional): A
Masks
object
containing
the
detection
masks.
probs(Probs, optional): A
Probs
object
containing
probabilities
of
each


class for classification task.


keypoints(Keypoints, optional): A
Keypoints
object
containing
detected
keypoints
for each object.
    speed(dict): A
    dictionary
    of
    preprocess, inference, and postprocess
    speeds in milliseconds
    per
    image.
names(dict): A
dictionary
of


class names.


path(str): The
path
to
the
image
file.
_keys(tuple): A
tuple
of
attribute
names
for non - empty attributes.
    .Did
you
mean: 'names'?