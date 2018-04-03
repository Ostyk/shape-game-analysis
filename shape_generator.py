# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

def shape_creator(Figure_type='triangle',l=10, step=100, spec=0.1,ideal=True,line_thickness=10,color='black'):
    fig=plt.figure(figsize=(3,3))
    if Figure_type == 'triangle':
        size = (np.random.randint(low=-8,high=-2), np.random.randint(low=10,high=13))
        if ideal is True:
            x1 = np.linspace(0,l,step)
            y1 = np.zeros(len(x1))
            x2 = np.linspace(l,int(l/2),step)
            y2 = np.linspace(0,int(l/2),step)
            x3 = np.linspace(int(l/2),0,step)
            y3 = np.linspace(int(l/2),0,step)
        elif ideal is False:
            test = np.linspace(-spec,spec,l) # range from which it randomly chooses from
            x1 = np.linspace(0,l,step)
            y1 = np.random.choice(test,size = len(x1))
            x2 = np.linspace(l,int(l/2),step) + np.random.choice(test,size = len(x1))
            y2 = np.linspace(0,int(l/2),step) + np.random.choice(test,size = len(x1))
            x3 = np.linspace(int(l/2),0,step) + np.random.choice(test,size = len(x1))
            y3 = np.linspace(int(l/2),0,step) + np.random.choice(test,size = len(x1))
        plt.plot(x1,y1,color,
                 x2,y2,color,
                 x3,y3,color,linewidth=line_thickness)
    elif Figure_type == 'square':
        size = (np.random.randint(low=10,high=17),np.random.randint(low=-8,high=-2))
        if ideal is True:
            x1 = np.linspace(0,l,step)
            y1 = np.zeros(len(x1))
            y2 = np.linspace(0,l,step)
            x2 = np.zeros(len(y2))
            x3 = np.linspace(0,l,step)
            y3 = np.ones(len(x3))*l
            y4 = np.linspace(0,l,step)
            x4 = np.ones(len(y2)) * l
        elif ideal is False:
            test = np.linspace(-spec,spec,l)
            x1 = np.linspace(0,l,step)
            y1 = np.random.choice(test,size = len(x1))
            y2 = np.linspace(0,l,step)
            x2 = np.random.choice(test,size = len(y2))
            x3 = np.linspace(0,l,step)
            y3 = np.ones(len(x3))*l + np.random.choice(test,size = len(x3))
            y4 = np.linspace(0,l,step)
            x4 = np.ones(len(y2)) * l + np.random.choice(test,size = len(x3))
        plt.plot(x1,y1,color,
        x2,y2,color,
        x3,y3,color,
        x4,y4,color,linewidth=line_thickness)
    elif Figure_type == 'circle':
        size = (np.random.randint(low=2,high=4),np.random.randint(low=-3,high=-1))
        r = np.sqrt(np.random.choice(np.linspace(0.5,1,20)))
        test = np.linspace(-spec,spec,l)
        theta = np.linspace(0, 2*np.pi, 100)
        x1 = r*np.cos(theta)
        if ideal is True:
            x2 = r*np.sin(theta)
        elif ideal is False:
            x2 = r*np.sin(theta) + np.random.choice(test,size = len(x1))
        plt.plot(x1,x2,color,linewidth=line_thickness)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.axes().set_aspect('equal')
    plt.xlim(size)
    plt.ylim(size)# with this we can alter the position of the figure on the image
    return fig


def data_generator(Figures=['squares','triangles','circles'],dateset_size = 10,test_size = 0.2 ):
    if not os.path.exists('data'):
        os.makedirs('data')
    os.chdir('data')
    if not os.path.exists('train'):
        os.makedirs('train')
    if not os.path.exists('test'):
        os.makedirs('test')

    train_size = 1 - test_size
    train_length = int(dateset_size*train_size/len(Figures)) #10*0.8 / 2 = 4
    test_lenth = int(dateset_size*test_size)


    os.chdir('train')
    for i in range(len(Figures)):
        if not os.path.exists(Figures[i]):
            print(Figures[i])
            os.makedirs(Figures[i])

    for i in range(len(Figures)): #generate training data
        os.chdir(Figures[i]) #enter square or triangle
        Figure_type = Figures[i][:-1] #square not squares, taking off the plural

        for i in range(int(train_length*0.95)):
                shape_creator(Figure_type=Figure_type.lower(),step=25,ideal=False)
                plt.savefig(Figure_type+"_"+str(i)+".png")
                plt.close()
        for i in range(int(train_length*0.05)):
                shape_creator(Figure_type=Figure_type.lower(),step=25,ideal=True)
                plt.savefig(Figure_type+"_"+str(i)+".png")
                plt.close()
        os.chdir("..") #exit square or triangle
    os.chdir("..") #exists back to data

    os.chdir('test')
    for i in range(len(Figures)):
        if not os.path.exists(Figures[i]):
            os.makedirs(Figures[i])

    for i in range(len(Figures)): #generate validation data
        os.chdir(Figures[i]) #enter square or triangle
        Figure_type = Figures[i][:-1] #square not squares, taking of the plural
        for i in range(int(test_lenth/len(Figures))):
                shape_creator(Figure_type=Figure_type.lower(),step=25,ideal=False)
                plt.savefig(Figure_type+"_"+str(i)+".png")
                plt.close()
        os.chdir("..") #exit square or triangle
    os.chdir("..") #exists back to data
    os.chdir("..")


dataset_size = int(input('please enter dataset size: '))
test_size = float(input('please enter size proportion of the validation set\n example: 0.2 on a dataset of 10 would mean that 2 inputs will be for testing\n please input your size: '))
data_generator(Figures=['squares','triangles','circles'],dateset_size=dataset_size,test_size=test_size)
print('dataset created as folder data')
