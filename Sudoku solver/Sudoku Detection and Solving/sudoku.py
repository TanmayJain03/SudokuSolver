import cv2
import numpy as np
import inspect, sys, re, operator
from model import Trainer
from solver import Solver
import imutils
from skimage.segmentation import clear_border

class Detector:
	def __init__(self):
		p = re.compile("stage_(?P<idx>[0-9]+)_(?P<name>[a-zA-Z0-9_]+)")

		self.stages = list(sorted(
		map(
			lambda x: (p.fullmatch(x[0]).groupdict()['idx'], p.fullmatch(x[0]).groupdict()['name'], x[1]),
			filter(
				lambda x: inspect.ismethod(x[1]) and p.fullmatch(x[0]),
				inspect.getmembers(self))),
		key=lambda x: x[0]))

		# For storing the recognized digits
		self.digits = [ [None for i in range(9)] for j in range(9) ]

	# Takes as input 9x9 array of numpy images
	# Combines them into 1 image and returns
	# All 9x9 images need to be of same shape
	def makePreview(images):
		assert isinstance(images, list)
		assert len(images) > 0
		assert isinstance(images[0], list)
		assert len(images[0]) > 0
		assert isinstance(images[0], list)

		rows = len(images)
		cols = len(images[0])

		cellShape = images[0][0].shape

		padding = 10
		shape = (rows * cellShape[0] + (rows + 1) * padding, cols * cellShape[1] + (cols + 1) * padding)
		
		result = np.full(shape, 255, np.uint8)

		for row in range(rows):
			for col in range(cols):
				pos = (row * (padding + cellShape[0]) + padding, col * (padding + cellShape[1]) + padding)

				result[pos[0]:pos[0] + cellShape[0], pos[1]:pos[1] + cellShape[1]] = images[row][col]

		return result


	# Takes as input 9x9 array of digits
	# Prints it out on the console in the form of sudoku
	# None instead of number means that its an empty cell
	def showSudoku(array):
		cnt = 0
		for row in array:
			if cnt % 3 == 0:
				print('+-------+-------+-------+')

			colcnt = 0
			for cell in row:
				if colcnt % 3 == 0:
					print('| ', end='')
				print('. ' if cell is None else str(cell) + ' ', end='')
				colcnt += 1
			print('|')
			cnt += 1
		print('+-------+-------+-------+')

	# Runs the detector on the image at path, and returns the 9x9 solved digits
	# if show=True, then the stage results are shown on screen
	# Corrections is an array of the kind [(1,2,9), (3,3,4) ...] which implies
	# that the digit at (1,2) is corrected to 9
	# and the digit at (3,3) is corrected to 4
	def run(self, path='assets/sudokus/sudoku1.jpg', show = False, corrections = []):
		self.path = path
		self.original = cv2.imread(path)

		self.run_stages(show)
		result = self.solve(corrections)


		if show:
			self.showSolved()
			cv2.waitKey(0)
			cv2.destroyAllWindows()

		return result

	# Runs all the stages
	def run_stages(self, show):
		results = [('Original', self.original)]

		for idx, name, fun in self.stages:
			image = fun().copy()
			results.append((name, image))

		if show:
			for name, img in results:
				cv2.imshow(name, img)
		

	# Stages
	# Stage function name format: stage_[stage index]_[stage name]
	# Stages are executed increasing order of stage index
	# The function should return a numpy image, which is shown onto the screen
	# In case you have 81 images of 9x9 sudoku cells, you can use makePreview()
	# to create a single image out of those
	# You can pass data from one stage to another using class member variables
	def stage_1_blur(self):

		image = cv2.cvtColor(self.original.copy(), cv2.COLOR_BGR2GRAY)
		image = cv2.GaussianBlur(image, (9,9), 0)

		self.image1 = image

		return image


	def stage_2_threshold(self):

		image = cv2.adaptiveThreshold(self.image1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 4)
		image = cv2.bitwise_not(image)

		self.image2 = image

		return image


	def stage_3_contours(self):

		# Finding largest rectangle using contours
		contour = cv2.findContours(self.image2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour = imutils.grab_contours(contour)
		contour = sorted(contour, key = cv2.contourArea, reverse = True)

		for c in contour:
			perimeter = cv2.arcLength(c, True)
			corners = cv2.approxPolyDP(c, 0.015 * perimeter, True)
			if len(corners) == 4:
				self.contour = corners
				break

		image = cv2.drawContours(self.original.copy(), [self.contour], -1, (0,255,0), 2)
		self.image3 = image

		return image


	def stage_4_perspective(self):

		corners = [(corner[0][0], corner[0][1]) for corner in self.contour]
		top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]

		# Finding width of sudoku puzzle
		width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
		width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
		width = max(int(width_A), int(width_B))

		# Finding height of sudoku puzzle
		height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
		height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
		height = max(int(height_A), int(height_B))

		dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
		ordered_corners = np.array(corners, dtype="float32")

		grid = cv2.getPerspectiveTransform(ordered_corners, dimensions)
		image = cv2.warpPerspective(self.original.copy(), grid, (width, height))
		image = cv2.flip(image, 1)
		self.image4 = image

		return self.image4

	def stage_5_sudoku(self):

		image = cv2.cvtColor(self.image4, cv2.COLOR_BGR2GRAY)

		# Finding cell width and height
		height = np.shape(image)[0] // 9
		width = np.shape(image)[1] // 9

		# Dividing the image into 9x9 array of images
		Sudoku = []
		for i in range(height, np.shape(image)[0] + 1, height):
			row = image[i - height: i]
			for j in range(width, np.shape(image)[1] + 1, width):
				 Sudoku.append([row[k][j - width: j] for k in range(len(row))])

		self.sudoku = []
		for i in range(0, len(Sudoku), 9):
			self.sudoku.append(Sudoku[i: i+9])

		for i in range(9):
			for j in range(9):
				self.sudoku[i][j] = cv2.resize(np.array(self.sudoku[i][j]), (28,28))

		return Detector.makePreview(self.sudoku)


	def digitPresent(self, cell):
		thresh = cv2.threshold(cell.copy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		thresh = clear_border(thresh)
		contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		contour = imutils.grab_contours(contour)

		if len(contour) == 0:
			return None

		c = max(contour, key = cv2.contourArea)
		mask = np.zeros(thresh.shape, dtype = "uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)

		(h,w) = thresh.shape
		Filled = cv2.countNonZero(mask) / float(w * h)

		if Filled < 0.03:
			return None

		digit = cv2.bitwise_and(thresh, thresh, mask = mask)

		return digit


	def stage_6_digits(self):

		t = Trainer()

		try:
			t.load_model()
		except:
			t.load_data()
			t.train()
			t.test()
	
		for i in range(9):
			for j in range(9):
				digit = self.digitPresent(self.sudoku[i][j]) 
				if digit is not None:
					self.sudoku[i][j] = digit
					self.digits[i][j] = t.predict(self.sudoku[i][j] / 255.0)

		return Detector.makePreview(self.sudoku)


	# Solve function
	# Returns solution
	def solve(self, corrections):
		# Only upto 3 corrections allowed
		assert len(corrections) <= 3

		# Apply the corrections
		for tup in corrections:
			self.digits[tup[0]][tup[1]] = tup[2]

		# Solve the sudoku
		self.answers = [[ self.digits[j][i] for i in range(9) ] for j in range(9)]
		s = Solver(self.answers)
		if s.solve():
			self.answers = s.digits
			return s.digits

		return [[None for i in range(9)] for j in range(9)]


	# Optional
	# Use this function to backproject the solved digits onto the original image
	# Save the image of "solved" sudoku into the 'assets/sudoku/' folder with
	# an appropriate name
	def showSolved(self):
		pass


if __name__ == '__main__':
	d = Detector()
	result = d.run('assets/sudokus/sudoku2.jpg', show=True, corrections = [(4,8,7)])
	print('Recognized Sudoku:')
	Detector.showSudoku(d.digits)
	print('\n\nSolved Sudoku:')
	Detector.showSudoku(result)
