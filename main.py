#this is the GUI Part of the project made using tkinter

from tkinter import *
from tkinter import filedialog;
from PIL import Image
import cv2
from PIL import EpsImagePlugin
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tkinter import messagebox as tmsg

#add plugin for reading  ghostscript file 
# !!!Note: this will not work if ghostscript is not installed in the system
EpsImagePlugin.gs_windows_binary='C:/Program Files/gs/gs9.53.3/bin/gswin64c'



class DigitRecognizer:
	def __init__(self):
		self.root= Tk()
		self.model = pickle.load(open("my_model.pickle",'rb')) #load the classifier model
		self.root.title("Digit Recognizer")
		self.root.resizable(False, False)
		self.penwidth = 86 #adjusted as per the maxmimum accuracy
		self.pencolor='black'
		self.canvas_height=600
		self.canvas_width=600
		self.canvas= None
		self.x = None
		self.y = None
		self.main()
		self.root.bind('<B1-Motion>', self.sketch)
		self.root.bind('<ButtonRelease-1>', self.reset)
		self.root.mainloop()

	def getImg(self): #get image from canvas object
		self.canvas.postscript(file='unprocessed.eps') #convert image to the eps file
		img = Image.open('unprocessed.eps')
		img.save('image.png','png')	#save file as png
		img = cv2.imread('image.png')
		resized_img =cv2.resize(img,(28,28),fx=0.1,fy=0.1) #resize the image into dimension: 28x28
		cv2.imwrite("resized.png", resized_img) #save resized image
		# resized_img.show()
		gray = cv2.imread("resized.png",0) #read image as grayscale
		gray = cv2.bitwise_not(gray) #invert the image tone , black<->white
		# cv2.imshow("input",gray)
		img_array = np.array(gray) #conver image to numpy ndarray object
		#print(img_array)
		# print(img_array.shape)
		self.detect_digit(img_array.reshape(784,))

	def detect_digit(self,img_array): #runs the model on the numpy array and shows the predicted digit in a window 
		#normalize
		# img_array = (255-img_array)/255
		# print(img_array)
		detected= (self.model.predict([img_array])[0])	#feed into model for prediction
		# plt.imshow(img_array.reshape((28,28)))
		# plt.show()
		tmsg.showinfo("Result",f"The number is {detected}")


	def reset(self, event): #reset mouse pointers for drawing on canvas
		self.x, self.y = None, None	
		
	def sketch(self, event): #draw on canvas
		if self.x and self.y:
			self.canvas.create_line(self.x,self.y ,event.x,event.y, width = self.penwidth ,fill = self.pencolor, capstyle=ROUND, smooth = True )
		self.x, self.y =event.x, event.y	
	
	def clearscreen(self): #clear canvas
		self.canvas.delete('all')
		#self.canvas.bg='white'	

	def main(self): #create the widgets
		self.root.geometry(f"{self.canvas_width}x{self.canvas_height+100}")	
		self.frame = Frame(self.root, bg ='gray', relief=SUNKEN, bd=5)
		self.frame.place(relx =0 ,rely =0, relwidth=1, relheight=1)

		self.canvas = Canvas(self.frame, bg='white', width=self.canvas_width,height=self.canvas_height)
		self.canvas.pack()
		# self.canvas.create_rectangle(50,50,100,100)
		# self.canvas.create_line(10,5,200,300)
		self.label= Label(self.frame, text='Please draw in the center of the canvas for accurate results', bg='gray', fg='black',justify="center", font=(15))
		self.label.place(relx =0.05 ,rely =0.88, relwidth=0.9, relheight=0.04)

		self.detect = Button(self.frame,bd=3, text ='Detect',command= lambda:self.getImg(),font=(15), bg = 'blue', fg='white')
		self.detect.place(relx =0.2 ,rely =0.935, relwidth=0.2, relheight=0.06)

		self.clear = Button(self.frame,bd=3, text ='Clear',command= lambda:self.clearscreen(), font=(15), bg ='red', fg='white')
		self.clear.place(relx =0.6 ,rely =0.935, relwidth=0.2, relheight=0.06)

if __name__=="__main__":
	DigitRecognizer()

#end of file





