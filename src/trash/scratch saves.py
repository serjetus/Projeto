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

if list_re[0][2] > list_re[1][2]:  # lado direito ou esquerdo do corpo esta visivel??
    if self.compare_circles(list_re[0][0], list_re[0][1]):  # ombro E branco?
        self.caracterics.append("REGATA")
    else:
        if self.compare_circles(list_re[2][0], list_re[2][1]):
            self.caracterics.append("CAMISA")
        else:
            self.caracterics.append("MANGA LONGA")
else:
    if self.compare_circles(list_re[1][0], list_re[1][1]):  # ombro D branco?
        self.caracterics.append("REGATA")
    else:
        if self.compare_circles(list_re[3][0], list_re[3][1]):
            self.caracterics.append("CAMISA")
        else:
            self.caracterics.append("MANGA LONGA")

if self.compare_circles(list_re[7][0], list_re[7][1]):
    self.caracterics.append('SHORTS')
else:
    self.caracterics.append('CALÇA')