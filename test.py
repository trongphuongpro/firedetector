import cv2
import numpy as np
import argparse

class FireDetector:
	def __init__(self):
		self.threshold = 25
		self.background = None
		self.accumWeight = 0.3
		self.foregroundImage = None
		self.foregroundAccumModel = None

		self.blockSize = 8
		self.b1 = 2
		self.b2 = 1
		self.colorLow = np.array([0, 0, 150])
		self.colorHigh = np.array([20, 255, 255])

		self.frameCount = 0
		self.intialized = False


	def update(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if self.background is None:
			self.background = gray.astype("float")

		cv2.accumulateWeighted(gray, self.background, self.accumWeight)

		if not self.intialized:
			self.frameCount += 1

			if self.frameCount == 10:
				self.intialized = True


	def detect(self, image):
		result = image.copy()

		if self.intialized:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			delta = cv2.absdiff(self.background.astype("uint8"), gray)
			thresh = cv2.threshold(delta, 
									self.threshold, 
									255, 
									cv2.THRESH_BINARY)[1]

			thresh = cv2.dilate(thresh, None, iterations=2)

			fg = cv2.bitwise_and(image, image, mask=thresh)

			self.foregroundImage = cv2.inRange(cv2.cvtColor(fg, cv2.COLOR_BGR2HSV), 
												self.colorLow, 
												self.colorHigh)

			if self.foregroundAccumModel is None:
				self.foregroundAccumModel = np.zeros(gray.shape, dtype="int32")

			self.foregroundAccumModel[self.foregroundImage == 255] += self.b1
			self.foregroundAccumModel[self.foregroundImage == 0] -= self.b2
			self.foregroundAccumModel = np.clip(self.foregroundAccumModel, 0, 255)

			blockImage = np.zeros(gray.shape, dtype="uint8")
			blockImage[self.foregroundAccumModel > 25] = 255

			
			for i in range(0, gray.shape[1], self.blockSize):
				for j in range(0, gray.shape[0], self.blockSize):
					if (np.count_nonzero(blockImage[j:j+self.blockSize, i:i+self.blockSize]) > self.blockSize**2//2):
						cv2.rectangle(result, (i,j), (i+self.blockSize,j+self.blockSize), (0,0,255), 1)

		return result


# class SmokeDetector:
# 	def __init__(self):
# 		self.prevFrame = None
# 		self.intialized = False


# 	def matchBlock(self, image):
# 		if self.intialized:
# 			currentFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 			motionMask = np.zeros((currentFrame.shape[0], currentFrame.shape[1]),
# 										dtype="uint8")
# 			F = np.zeros((3,), "float")
			
# 			for i in range(8, self.prevFrame.shape[1] - 8, 8):
# 				for j in range(8, self.prevFrame.shape[0], 8):
# 					M = []
# 					M.append(np.minimum(self.prevFrame[j:j+8, i:i+8], currentFrame[j-8:j, i-8:i]).min())
# 					M.append(np.maximum(self.prevFrame[j:j+8, i:i+8], currentFrame[j-8:j, i-8:i]).max())
# 					M.append(np.minimum(self.prevFrame[j:j+8, i:i+8], currentFrame[j-8:j, i:i+8]).min())
# 					M.append(np.maximum(self.prevFrame[j:j+8, i:i+8], currentFrame[j-8:j, i:i+8]).max())
# 					M.append(np.minimum(self.prevFrame[j:j+8, i:i+8], currentFrame[j-8:j, i+8:i+16]).min())
# 					M.append(np.maximum(self.prevFrame[j:j+8, i:i+8], currentFrame[j-8:j, i+8:i+16]).max())

# 					F[0] = M[0] / M[1]
# 					F[1] = M[2] / M[3]
# 					F[2] = M[4] / M[5]

		
# 					idx = np.argmax(F)
					
# 					motionMask[j-8:j, i+idx*8-8:i+idx*8] = 255

# 			self.prevFrame = currentFrame
# 			return motionMask

# 		return None

# 	def update(self, image):
# 		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		if not self.intialized:
# 			self.prevFrame = gray
# 			self.intialized = True

class SmokeDetector:
	def __init__(self):
		self.threshold = 25
		self.background = None
		self.accumWeight = 0.3
		self.foregroundImage = None
		self.foregroundAccumModel = None

		self.blockSize = 4
		self.b1 = 5
		self.b2 = 1

		self.a = 15
		self.K1 = 50
		self.K2 = 230

		self.kernel = np.ones((5, 5))

		self.frameCount = 0
		self.intialized = False


	def update(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		if self.background is None:
			self.background = gray.astype("float")

		cv2.accumulateWeighted(gray, self.background, self.accumWeight)

		if self.frameCount < 10:
			self.frameCount += 1

		if self.frameCount == 10:
			self.intialized = True


	def detect(self, image):
		result = image.copy()

		if self.intialized:
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

			delta = cv2.absdiff(self.background.astype("uint8"), gray)
			thresh = cv2.threshold(delta, 
									self.threshold, 
									255, 
									cv2.THRESH_BINARY)[1]

			thresh = cv2.dilate(thresh, None, iterations=2)

			fg = cv2.bitwise_and(image, image, mask=thresh)

			m = np.maximum(np.maximum(fg[...,0], fg[...,1]), fg[...,2])
			n = np.minimum(np.minimum(fg[...,0], fg[...,1]), fg[...,2])
			I = np.sum(fg, axis=2)/3

			cond_1 = np.where(m-n < self.a, True, False)
			cond_2 = np.logical_and(I >= self.K1, I <= self.K2)

			self.foregroundImage = np.zeros(gray.shape, dtype="uint8")
			self.foregroundImage[np.logical_and(cond_1, cond_2)] = 255
			#cv2.imshow("fg", self.foregroundImage)
			if self.foregroundAccumModel is None:
				self.foregroundAccumModel = np.zeros(gray.shape, dtype="int32")

			self.foregroundAccumModel[self.foregroundImage == 255] += self.b1
			self.foregroundAccumModel[self.foregroundImage == 0] -= self.b2
			self.foregroundAccumModel = np.clip(self.foregroundAccumModel, 0, 255)

			blockImage = np.zeros(gray.shape, dtype="uint8")
			blockImage[self.foregroundAccumModel > 25] = 255

			
			for i in range(0, gray.shape[1], self.blockSize):
				for j in range(0, gray.shape[0], self.blockSize):
					if (np.count_nonzero(blockImage[j:j+self.blockSize, i:i+self.blockSize]) > self.blockSize**2//2):
						cv2.rectangle(result, (i,j), (i+self.blockSize,j+self.blockSize), (0,255,0), 1)

		return result

fd = FireDetector()
sd = SmokeDetector()

ap = argparse.ArgumentParser()
ap.add_argument("video")
args = vars(ap.parse_args())

video = cv2.VideoCapture("FireClips\\" + args["video"])
frameCount = 0

while True:
	grab, frame = video.read()
	if not grab:
		print("No frame!")
		break

	smokeResult = sd.detect(frame)
	fireResult = fd.detect(frame)
	sd.update(frame)
	fd.update(frame)

	if smokeResult is not None:
		cv2.imshow("smoke", smokeResult)
		cv2.imshow("fire", fireResult)
		#cv2.imshow("frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == 27:
		break

video.release()
cv2.destroyAllWindows()