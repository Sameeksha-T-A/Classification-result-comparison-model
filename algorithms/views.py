from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render
import multiprocessing
from django.http import HttpResponse, HttpResponseRedirect
import subprocess
import distutils.dir_util
from django.conf import settings
import os, datetime

from django.core.mail import send_mail
from django.core.mail import EmailMessage
#from pred import *
#from . import pred

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

import datetime
import os
from io import BytesIO
import base64

def index(request):
	if request.POST and request.FILES:
	
		try:
			global filename, id, emailid
			emailid = request.POST['emailid']
			filename = request.FILES['csv_file']

			id = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
			#enter your directory path after creating media and result folders
			dir_path = os.path.join('C:\\Users\\Gumbi\\Desktop\\alg_comparison\\media\\result', id)

			folder = os.mkdir(dir_path)
			dataset = pd.read_csv(filename)

			X = dataset.iloc[:,[2,3]].values
			y = dataset.iloc[:,4].values

			from sklearn.model_selection import train_test_split
			X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.25, random_state = 0)

			from sklearn.preprocessing import StandardScaler
			sc_X = StandardScaler()
			X_train = sc_X.fit_transform(X_train)
			X_test = sc_X.transform(X_test)


			#Decision Tree
			from sklearn.tree import DecisionTreeClassifier
			classifier = DecisionTreeClassifier(criterion = 'entropy',random_state = 0)
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)

			from sklearn.metrics import classification_report
			a=(classification_report(y_test, y_pred))

			with open(os.path.join(dir_path,"decision" + ".log"), 'w') as file:
				file.write("\n -----DECISION TREE-----\n")

			with open(os.path.join(dir_path,"decision" + ".log"), 'a') as file:
				for line in a:
					file.write(str(line))
	

			#fp = open("media/" + "result/" + str(id) + "/" + str(id) + ".log", 'a')
			#file.write("\n DECISION TREE SUCCESSFULLY COMPLETED\n\n\n")
			#fp.close()

			from matplotlib.colors import ListedColormap
			X_set, y_set=X_test,y_test
			X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

			plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
			plt.xlim(X1.min(),X1.max())
			plt.ylim(X2.min(),X2.max())
			

			for i,j in enumerate(np.unique(y_set)):
				plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
			plt.title('Decision tree classifier(Testing set)')
			plt.xlabel('X axis')
			plt.ylabel('Y axis')
				
			sample_file_name = "decision-test-result"
			plt.savefig(dir_path + "/" + sample_file_name)

			buffer=BytesIO()
			plt.savefig(buffer,format='png')
			buffer.seek(0)
			image_png1 = buffer.getvalue()
			buffer.close()
			plt.cla()

			graphic1 = base64.b64encode(image_png1)
			graphic1 = graphic1.decode('utf-8')


			#Support Vector Machine			
			from sklearn.svm import SVC
			classifier = SVC(kernel = 'linear', random_state = 0)
			classifier.fit(X_train, y_train)

			y_pred = classifier.predict(X_test)

			from sklearn.metrics import classification_report
			b=(classification_report(y_test, y_pred))

			with open(os.path.join(dir_path,"svm" + ".log"), 'w') as file:
				file.write("\n -----SVM-----\n")
			

			with open(os.path.join(dir_path,"svm" + ".log"), 'a') as file:
				for line in b:
					file.write(str(line))
				


			from matplotlib.colors import ListedColormap
			X_set, y_set = X_test, y_test
			X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     		np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
			plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            				alpha = 0.75, cmap = ListedColormap(('blue', 'cyan')))
			plt.xlim(X1.min(), X1.max())
			plt.ylim(X2.min(), X2.max())
			

			for i, j in enumerate(np.unique(y_set)):
				plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c = ListedColormap(('blue', 'cyan'))(i), label = j)
			plt.title('SVM (Test set)')
			plt.xlabel('X axis')
			plt.ylabel('Y axis')


			sample_file_name = "svm-test-result"
			plt.savefig(dir_path + "/" + sample_file_name)

			buffer=BytesIO()
			plt.savefig(buffer,format='png')
			buffer.seek(0)
			image_png2 = buffer.getvalue()
			buffer.close()
			plt.cla()

			graphic2 = base64.b64encode(image_png2)
			graphic2 = graphic2.decode('utf-8')

			#return render(request, 'progress.html',{'graphic':graphic2})


			#Random Forest
			from sklearn.ensemble import RandomForestClassifier
			classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0 )
			classifier.fit(X_train,y_train)
			y_pred = classifier.predict(X_test)

			from sklearn.metrics import classification_report
			c=(classification_report(y_test, y_pred))


			with open(os.path.join(dir_path,"random" + ".log"), 'w') as file:
				file.write("\n -----RANDOM FOREST-----\n")
			

			with open(os.path.join(dir_path,"random" + ".log"), 'a') as file:
				for line in c:
					file.write(str(line))
			


			from matplotlib.colors import ListedColormap
			X_set, y_set=X_test,y_test
			X1,X2 = np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:,0].max()+1,step=0.01),np.arange(start=X_set[:,1].min()-1,stop=X_set[:,1].max()+1,step=0.01))

			plt.contourf(X1,X2, classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('magenta','yellow')))
			plt.xlim(X1.min(),X1.max())
			plt.ylim(X2.min(),X2.max())



			for i,j in enumerate(np.unique(y_set)):
				plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('magenta','yellow'))(i),label=j)
			plt.title('Random Forest classifier(Testing set)')
			plt.xlabel('X axis')
			plt.ylabel('Y axis')

			sample_file_name = "randomforest-test-result"
			plt.savefig(dir_path + "/" + sample_file_name)

			buffer=BytesIO()
			plt.savefig(buffer,format='png')
			buffer.seek(0)
			image_png3 = buffer.getvalue()
			buffer.close()
			plt.cla()

			graphic3 = base64.b64encode(image_png3)
			graphic3 = graphic3.decode('utf-8')

			#return render(request, 'progress.html',{'graphic':graphic3})

			#fp = open("media/" + "result/" + str(id) + "/" + str(id) + ".log", 'a')
			#fp.write("\nCLASSIFICATION DONE!!")
			#fp.close()

			f1 = open("media/" + "result/" + str(id) + "/" + "decision" + ".log", 'r')
			f2 = open("media/" + "result/" + str(id) + "/" + "svm" + ".log", 'r')
			f3 = open("media/" + "result/" + str(id) + "/" + "random" + ".log", 'r')
			
			file1=""
			file2=""
			file3=""
			refresh =True

			for i in f1:
				file1 = "      "+ file1 +  "\n " + i + "\n"
				for i in f2:
					file2 = "      "+ file2 +  "\n " + i + "\n"
					for i in f3:
						file3 = "      "+ file3 +  "\n " + i + "\n"			
			return render(request,"progress.html",context = {'file1':file1, 'file2':file2,'file3':file3,
				'graphic1':graphic1,'graphic2':graphic2,'graphic3':graphic3})

	
		except Exception as e:
			print(e)
			print(request.FILES)
			return HttpResponse("""<h3> Oooppss!! Errorr! </h3>""")

	return render(request, 'index.html');
'''
def progress(request):
	fp = open("media/" + "result/" + str(id) + "/" + str(id) + ".log", 'r')
	file=""
	refresh =True
	for i in fp:
		if i == "CLASSIFICATION DONE":
			refresh = False
		file = "      "+ file +  "\n " + i + "\n"
	return render(request,"progress.html",{'file':file})

'''




