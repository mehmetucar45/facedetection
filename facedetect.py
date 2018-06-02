# by Mehmet Ucar
# Face Recognition Library
# TAMU RET Summer 2017
#!/usr/bin/python3


from Tkinter import *
import Tkinter,tkFileDialog,tkMessageBox
from yattag import Doc
from PIL import ImageTk, Image
import easygui as eg
import os
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from datetime import datetime
import webbrowser

doc, tag, text = Doc().tagtext()
doc.asis('<!DOCTYPE html>')
class Checkbar(Frame):
   def __init__(self, parent=None, picks=[], side=LEFT, anchor=W):
      Frame.__init__(self, parent)
      self.vars = []
      for pick in picks:
         var = IntVar()
         chk = Checkbutton(self, text=pick, variable=var, justify=LEFT, font=("Helvetica", 10), bd=1,  bg="pink")
	 #chk1 = Checkbutton(self, text=pick, variable=var, justify=LEFT, font=("Helvetica", 10), bd=1,  bg="pink")
         chk.pack(side=side, anchor=anchor, expand=YES, fill=X)
	 #chk1.pack(side=side, anchor=anchor, expand=YES, fill=X)
         self.vars.append(var)
   def state(self):
      return map((lambda var: var.get()), self.vars)
if __name__ == '__main__':
   root = Tk()
   root.title('M Ucar - TAMU RET 2017')
   #root.geometry("550x300+300+150")
   #root.resizable(width=True, height=True)
   #root.iconbitmap(default='icon.ico')
   Titleimg=ImageTk.PhotoImage(Image.open("RET_Title.png"))#title image
   panel = Label(root, image = Titleimg, bg = "white", activebackground="green")
   panel.pack(side = "top", fill = "both", expand = "no", )
   TitleText = Label(root, text="\n\t\t   MAKER : Face Detection Library to Teach Algorithm Basics in Python\t\t   \nSummer 2017", font=("Helvetica", 14), bg="white")
   TitleText.pack(fill=X)
   TitleText2 = Label(root, text="by Mehmet (\"Matt\") Ucar - Supervisor: Dr. Sheng-Jen (\"Tony\") Hsieh", font=("Helvetica", 10), bg="white")
   TitleText2.pack(fill=X)
   FaceParts = Checkbar(root, ['Mouth','Right Eyebrow', 'left Eyebrow', 'Right Eye','Left Eye', 'Nose', 'Jaw'], )
   #Scroller set
   #Algorithm = Checkbar(root, ['DLIB Frontal Face Detector','HaarCascades'])
   #AlgText=Label(root, text="\nSelect the name of the face detection algorithm below:", font=("Helvetica",12))
   #AlgText.pack()
   #Algorithm.pack(side=TOP)
   #Algorithm.config(relief=GROOVE, bd=2)
   Alg=1
   #Alg = IntVar()
   #Radiobutton(root, text="DLIB Frontal Face Detector",font=("Helvetica",12), variable=Alg, value=1, indicatoron=0).pack(side= TOP,fill=X, anchor=W)
   #Radiobutton(root, text="HaarCascades",font=("Helvetica",12), variable=Alg, value=2,indicatoron=0).pack(side=TOP,fill=X, anchor=W)
   '''
   def selection():
   	question = "STEP 1 :\nSelect the parts of the faces to be Extracted using DLIB below"
   	title = "STEP 1"
   	listOfOptions = ['Mouth','Right Eyebrow', 'left Eyebrow', 'Right Eye','Left Eye', 'Nose', 'Jaw']
   	choice = eg.multchoicebox(question , title, listOfOptions)
	sprint (choice)
	return choice
   '''
   FaceText=Button(root, text="STEP1: Select the parts of the faces to be Extracted using DLIB below",pady=10, font=("Helvetica",14), bg="pink")
   FaceText.pack(fill=X)
   FaceParts.pack(fill=X)   
   FaceParts.config(relief=GROOVE, bd=1,  bg="pink")
   val = ["","0"]
   def chooseimage():
	filename = tkFileDialog.askopenfilename(title='Choose an Image file',multiple=False)
	if filename:
		with open(filename) as file:
			#imgdir = Image.open(filename)
			#img=ImageTk.PhotoImage(img)
			#img = imutils.resize(img, width=500)
			imgdir=filename
			print 'The image is selected as :',imgdir
			#panel1 = Label(root, image = img)
   			#panel1.pack(side = "top", fill = "both")
			#Notice1=Label(root, text="Your image is: %s" %imgdir, font=("Helvetica",8) )
   			#Notice1.pack()
			val[0]=imgdir
			val[1]="1";
			print ('Value of val :  %s' %val)
			tkMessageBox.showinfo("Your image is copied", "Congrats, you successfully selected your image file.") 
			step2.config(bg="green")
			return list(val)

   step2=Button(root, text='STEP 2: Choose the image file to process', font=("Helvetica",14), bg="pink", command=chooseimage,pady=10).pack(side=TOP, fill=X)    
   def allstates():
	if(val[1]=="0"):
		tkMessageBox.showinfo("Please Correct", "Please choose your input image file first")
	elif(val[1]=="1"):	
		parts=list(FaceParts.state())
		print(parts, Alg) #Alg.get
		doc, tag, text = Doc().tagtext()
		doc.asis('<!DOCTYPE html>')
		with tag('html'):
			doc._append('<head><link rel="stylesheet" type="text/css" href="theme.css"></head> ')
			with tag('title'):
				text('Report is generated')
			with tag('body'):
				with tag('div' ,id='title'):
					text('An Implementation of Face Detection Library to Teach Algorithm Basics in OpenCV and Python   \nSummer 2017')
				with tag('div', id='logo'): 
	   				doc.stag('img', src='RET_Title.png', klass="logo")
			
				##run if the dlib library is selected...
				if(Alg==1): #Alg.get()==1
					imgdir=val[0]
					with tag('h1'):
						text('Your input image is the following')
					with tag('div', id='photo-container'):
	    					doc.stag('img', src='%s' %imgdir, klass="photo")
					# construct the shape predictor algorithm.
					print ('DLIB Frontal face recognition algorithm is selected.')
					with tag('h2'):
						text('DLIB Frontal face recognition algorithm is selected.')
					args = ("-p shape_predictor_68_face_landmarks.dat")
					# initialize dlib's face detector (HOG-based) and then create
					# the facial landmark predictor
					detector = dlib.get_frontal_face_detector()
					predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

					# load the input image, resize it, and convert it to grayscale
					
					image = cv2.imread(imgdir)
					image = imutils.resize(image, width=500)
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					doc._append("<br><hr>")					
					print ('....\tProcessing %s\t....\n' %imgdir)
					with tag('h4', id="center"):
						text('....\tProcessing %s\t....' %imgdir)
					#result='....\tProgram started processing %s\t....\n'%imgdir
					# detect faces in the grayscale image
					rects = detector(gray, 1)

					if(rects):
						print ('Number of faces detected in the image: %d\n' %len(rects))
						#result=result+'Number of faces detected in the image: %d\n' %(len(rects))
						with tag('h3'):
							text('Number of faces detected in the image: %d' %len(rects))
						# clone the original image so we can draw on it, then
						# draw rectangles on the image
						cloneRect = image.copy()	
						# loop over the face detections
						fno=0
						for (i, rect) in enumerate(rects):
							# determine the facial landmarks for the face region, then
							# convert the landmark (x, y)-coordinates to a NumPy array
							shape = predictor(gray, rect)
							shape = face_utils.shape_to_np(shape)		
							fno=fno+1
							print('STARTED processing FACE #%d' %fno)
							doc._append("<br><hr>")						
							with tag('h2', id='faces'):
								text('STARTED processing FACE #%d' %fno)
							#result=result+'STARTED processing FACE #%d' %fno
							# loop over the face parts individually
							k=-1;
							for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
								k=k+1							
								if(parts[k]==1):
									# clone the original image so we can draw on it, then
									# display the name of the face part on the image
									clone = image.copy()
									cv2.putText(clone, str(fno)+"_"+name.upper()+' Circled', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
										0.7, (0, 255, 0), 2)
									# loop over the subset of facial landmarks, drawing the
									# specific face part
									for (x, y) in shape[i:j]:
										cv2.circle(clone, (x, y), 1, (0, 255,0 ), 2)

									# extract the ROI of the face region as a separate image
									(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
									roi = image[y:y + h, x:x + w]
									roi = imutils.resize(roi, height=150, inter=cv2.INTER_CUBIC)
									# add text to resized ROI image
									cv2.putText(roi, str(fno)+"_"+name.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
										0.7, (0, 255, 0), 1)
									# show the particular face part
									#cv2.imshow("cropped %s" %name, roi)
									if (cv2.imwrite("allfaceparts/%d__cropped %s.jpg" %(fno,name), roi)):
										print('\t%d_%s \tDetected, circled and saved.' % (fno,name.upper().ljust(15)))
										#result=result+'\t%d_%s \tDetected, circled and saved.\n' % (fno,name.upper().ljust(15))
										with tag('div'):
											text('%s Detected, circled and saved for Face #%d.' % (name.upper().ljust(15),fno))
										with tag('div', id='photo-container'):
		    									doc.stag('img', src='allfaceparts/%d__cropped %s.jpg' %(fno,name), klass="faceparts-big")							
										#cv2.imshow("Image %s" %name, clone)
									if (cv2.imwrite("allfaceparts/%d_image %s.jpg" %(fno,name), clone)):
										print('\t%d_%s \tCropped and saved.' %(fno,name.upper().ljust(15)))
										with tag('div'):
											text('%s Cropped and saved for Face #%d.' % (name.upper().ljust(15),fno))
										with tag('div', id='photo-container'):
		    									doc.stag('img', src='allfaceparts/%d_image %s.jpg' %(fno,name), klass="faceparts")
										#result=result+'\t%d_%s \tCropped and saved.\n' %(fno,name.upper().ljust(15))
									#cv2.waitKey(0)
								else:
									print('Skipped analyzing %s on the face %d' %(name,fno))
									with tag('div'):
										text('Skipped analyzing %s on the face %d' %(name,fno))
							# convert dlib's rectangle to a OpenCV-style bounding box
							# [i.e., (x, y, w, h)], 
							(x, y, w, h) = face_utils.rect_to_bb(rect)		
		
							#crop the detected face and save it	 	
							crop_img= cloneRect[y:y+h, x:x+w]

							#Resize the cropped image
							crop_img = imutils.resize(crop_img, height=450, inter=cv2.INTER_CUBIC)
							# add text to resized ROI image
							cv2.putText(crop_img, "FACE_#"+str(fno), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
								0.7, (0, 255, 0), 1)
		
							if(cv2.imwrite('allfaceparts/Face_%d.jpg' %fno,crop_img)):
								print('Face #%d is cropped and saved.' %fno)		
								with tag('div'):
									text('Face #%d is cropped and saved.' %fno)
								with tag('div', id='photo-container'):
	    								doc.stag('img', src='allfaceparts/Face_%d.jpg' %fno, klass="fullface")
								#result=result+'Face #%d is cropped and saved.' %fno
							cv2.rectangle(cloneRect, (x, y), (x + w, y + h), (0, 255, 0), 2)		
							# show the face number
							cv2.putText(cloneRect, "Face #%d" %fno, (x - 10, y - 10),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
						 
							# loop over the (x, y)-coordinates for the facial landmarks
							# and draw them on the image
							for (x, y) in shape:
								cv2.circle(cloneRect, (x, y), 1, (0, 0, 255), -1)
		
							#cv2.imwrite("allfaceparts/%d_Complete face.jpg" %fno,cloneRect)
							print('COMPLETED processing FACE #%d.\n' %fno)
							with tag('h3', id='faces'):
								text('COMPLETED processing FACE #%d' %fno)
							#result=result+'COMPLETED processing FACE #%d.\n' %fno		
							#cv2.waitKey(0)
						cv2.imwrite("allfaceparts/Faces_Complete.jpg",cloneRect)
						print('All detected faces are shown on complete image.')	
						#result=result+'All detected faces are shown on complete image.'
						doc._append("<br><hr>")						
						with tag('h3'):
							text('All detected faces are shown on complete image.')
						with tag('div', id='photo-container'):
	    						doc.stag('img', src='allfaceparts/Faces_Complete.jpg', klass="fullface")
						doc._append("<br><hr>")						
					else:
						print ('We found No faces in the image to analyze all face parts.')
						#result=result+'We found No faces in the image to analyze all face parts.'
						doc._append("<br><hr>")						
						with tag('h4'):
							text('We found No faces in the image to analyze all face parts.')
					print ('\n....\tProgram completed processing %s"\t....' %imgdir)
					with tag('div'):
						text('....\tProgram completed processing image')
					doc._append("<br><hr>")					
					#result=result+'\n....\tProgram completed processing "sample.jpg"\t....'	
					#FaceText=Label(root, text=result, font=("Helvetica",6))
				   	#FaceText.pack(side=LEFT)
				##end if	      
				##run if the dlib library is selected...
				elif(Alg.get()==2):
					'''				
					print ('Haarcascade algorithm is selected.')
					with tag('h4'):
							text('Haarcascade algorithm is selected.')
					def draw_rects(img, rects, color):
	    					for x1, y1, x2, y2 in rects:
							cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
					def detect(img, cascade):
	    					rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
	    					if len(rects) == 0:
							return []
	    					rects[:,2:] += rects[:,:2]
	    					return rects
					cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalface_alt.xml")
					nested_fn  = args.get('--nested-cascade', "haarcascades/haarcascade_eye.xml")
					cascade = cv2.CascadeClassifier(cascade_fn)
					nested = cv2.CascadeClassifier(nested_fn)
					'''
				else:
					tkMessageBox.showinfo("Please Correct", "Please choose an algorithm to continue")
					with tag('h4'):
						text('Please choose a valid file to generate report')
		result = doc.getvalue()
		time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		report= open("Report_%s.html" %time,"w+")
		report.write(result)
		report.close()
		def callback(event):
    			webbrowser.open_new(r"Report_%s.html" %time)
		done=Label(root, text="STEP 4: Click to open the Report_%s file " %time, font=("Helvetica",12), pady=10, bg="pink", fg="blue", cursor="hand2" )
	   	done.pack(fill=X)
		done.bind("<Button-1>", callback)
		tkMessageBox.showinfo("Done", "Done your report file is saved.")		
		#webbrowser.open_new(r"Report_%s.html" %time)
		#root.quit()
		#print (result+'   heyy')
		#Report_%s.html" %time	
		
	#end of if(imgdir)
	
   	
   #end of allstates
   Button(root, text='STEP 3: RUN',  font=("Helvetica",14), justify=CENTER, bg="pink", pady=10, command=allstates).pack(side=TOP, fill=X)
   Button(root, text='QUIT',  font=("Helvetica",14),justify=CENTER,  bg="pink",command=root.quit, pady=10).pack(side=TOP, fill=X)

   #end the code
   root.mainloop()
