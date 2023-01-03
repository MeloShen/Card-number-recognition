from imutils import contours
import numpy as np
import argparse
import cv2
import myutils

# Set parameters
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-t", "--template", required=True,
	help="path to template OCR-A image")
args = vars(ap.parse_args())

# def the function for display
def cv_show(name,img):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Read a template image
img = cv2.imread(args["template"])
cv_show('img',img)

# Grayscale
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref',ref)

# Binary image
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref',ref)

# Calculated contour
# The parameters accepted by the cv2.findContours() function are binary graph,
# i.e. black and white (not gray),cv2.RETR_EXTERNAL only detects the outer contour,
# and cv2.CHAIN_APPROX_SIMPLE only retains the endpoint coordinates
# Each element in the list returned is an outline in the image
ref_, refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,refCnts,-1,(0,0,255),3)
cv_show('img',img)
# Sort from left to right, top to bottom
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# Walk through each contour
for (i, c) in enumerate(refCnts):
	# Calculate the enclosing rectangle and resize it to the appropriate size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))
	# Each number corresponds to each template
	digits[i] = roi

# Initialize the convolution kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Read input image, preprocess
image = cv2.imread(args["image"])
cv_show('image', image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# Top hat operation to highlight brighter areas
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)
#ksize=-1 is the same thing as 3 times 3
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)

gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

print (np.array(gradX).shape)
cv_show('gradX',gradX)

# Connect numbers together by closing operation (first expansion, then corrosion)
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX',gradX)

#THRESH_OTSU will automatically find a proper threshold, suitable for double peaks.
# Set the threshold parameter to 0
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh',thresh)

# One more close operation
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh',thresh)

# Compute contour
thresh_, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)
locs = []

# Walk the outline
for (i, c) in enumerate(cnts):
	# Calculate rectangle
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
# Select the appropriate area, depending on the actual task, this is usually a group of four numbers
	if ar > 2.5 and ar < 4.0:

		if (w > 40 and w < 55) and (h > 10 and h < 20):
			# Keep the ones that fit the bill
			locs.append((x, y, w, h))
# Sort the matching Outlines from left to right
locs = sorted(locs, key=lambda x:x[0])
output = []
# Walk through the numbers in each outline
for (i, (gX, gY, gW, gH)) in enumerate(locs):
	# initialize the list of group digits
	groupOutput = []
# Extract each group according to the coordinates
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	cv_show('group',group)
# Preprocessing
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	cv_show('group',group)
	#Calculate the outline of each group
	group_,digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	digitCnts = contours.sort_contours(digitCnts,
		method="left-to-right")[0]

# Calculate each number in each group
	for c in digitCnts:
	#Find the outline of the current value and resize it to the appropriate size
		(x, y, w, h) = cv2.boundingRect(c)
		roi = group[y:y + h, x:x + w]
		roi = cv2.resize(roi, (57, 88))
		cv_show('roi',roi)
		#Calculate the match score
		scores = []
		# Calculate each score in the template
		for (digit, digitROI) in digits.items():
			# Template matching
			result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# To get the most appropriate number
		groupOutput.append(str(np.argmax(scores)))
	# Draw
	cv2.rectangle(image, (gX - 5, gY - 5),
		(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
	cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
	# get result
	output.extend(groupOutput)

# print result
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)


