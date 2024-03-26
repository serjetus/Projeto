[[[4.783112525939941, 15.816670417785645, 0.5990004539489746],
  [5.8988728523254395, 13.678866386413574, 0.7298263311386108],
  [5.725320816040039, 13.446148872375488, 0.1288100779056549],
  [9.753800392150879, 13.775533676147461, 0.8805581331253052],
  [11.243636131286621, 12.689309120178223, 0.07875635474920273],
  [14.216670989990234, 28.809755325317383, 0.9409533143043518],
  [13.312222480773926, 27.050457000732422, 0.623717725276947],
  [13.492172241210938, 52.18115234375, 0.8206101059913635],
  [12.662620544433594, 49.52824783325195, 0.18340015411376953],
  [8.112222671508789, 68.04337310791016, 0.6811478734016418],
  [6.375480651855469, 67.7111587524414, 0.21046924591064453],
  [16.12566566467285, 67.68976593017578, 0.9371213316917419],
  [15.86115837097168, 66.68701171875, 0.8552242517471313],
  [14.487728118896484, 96.31819152832031, 0.8892371654510498],
  [14.636730194091797, 94.81464385986328, 0.7815163135528564],
  [18.64664649963379, 119.42422485351562, 0.48676273226737976],
  [16.39482879638672, 118.27100372314453, 0.38523274660110474]]]

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