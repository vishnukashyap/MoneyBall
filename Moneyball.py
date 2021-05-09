#	Predicting the number of wins in baseball (MLB) using the moneyball dataset.


#	Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.autograd import Variable

#	Dataset

df = pd.read_csv("baseball.csv")

df['RD'] = df['RS'] - df['RA']

newdf = df[ df['Year'] < 2002 ]

newdf = newdf.dropna(subset=['OOBP','OSLG'])

#	class for the Model

class Moneyball_LinearRegression(torch.nn.Module):

	def __init__(self):

		super(Moneyball_LinearRegression, self).__init__()

		self.linear = torch.nn.Linear(2,1)

	def forward(self, inputs):

		output = self.linear(inputs)

		return output


class Wins_Regression(torch.nn.Module):

	def __init__(self):

		super(Wins_Regression, self).__init__()

		self.linear = torch.nn.Linear(1,1)

	def forward(self, inputs):

		output = self.linear(inputs)

		return output

#	Main function

if __name__ == '__main__':

    #	Models for prediction

    RS_Model = Moneyball_LinearRegression()

    RA_Model = Moneyball_LinearRegression()

    Wins_Model = Wins_Regression()

    # 	Loss Function

    criterion_RS = torch.nn.L1Loss()
    
    criterion_RA = torch.nn.L1Loss()
    
    criterion_Win = torch.nn.L1Loss()

    #	Optimizer functions

    RS_optimizer = torch.optim.SGD(RS_Model.parameters(), lr=0.1, momentum=0.92)

    RA_optimizer = torch.optim.SGD(RA_Model.parameters(), lr=0.1, momentum=0.92)

    Win_optimizer = torch.optim.SGD(Wins_Model.parameters(), lr=0.1, momentum=0.98)

    #	Input and target DATA variables

    Input_arrayRS = []

    Input_arrayRA = []

    Input_arrayWin = []
    
    Output_arrayRS = []
    
    Output_arrayRA = []
    
    Output_arrayWin = []

    Temp_RS = np.array([newdf['OBP'],newdf['SLG']])

    Temp_RA = np.array([newdf['OOBP'],newdf['OSLG']])
    
    Temp_Win = np.array(newdf['RD'])
    
    TempRS = np.array(newdf['RS'])
    
    TempRA = np.array(newdf['RA'])
    
    TempWin = np.array(newdf['W'])

    for i in range(newdf.shape[0]):

        Input_arrayRS.append(list(Temp_RS[:,i]))

        Input_arrayRA.append(list(Temp_RA[:,i]))
        
        Input_arrayWin.append([Temp_Win[i]])
        
        Output_arrayRS.append([TempRS[i]])
        
        Output_arrayRA.append([TempRA[i]])
        
        Output_arrayWin.append([TempWin[i]])


    InputTensor_RS = Variable(torch.Tensor(Input_arrayRS))
    
    InputTensor_RA = Variable(torch.Tensor(Input_arrayRA))
    
    InputTensor_W = Variable(torch.Tensor(Input_arrayWin))     
    
    OutputTensor_RS = Variable(torch.Tensor(Output_arrayRS))
    
    OutputTensor_RA = Variable(torch.Tensor(Output_arrayRA))
    
    OutputTensor_Win = Variable(torch.Tensor(Output_arrayWin))


    Number_Epoch = 100000
    
    for epoch in range(Number_Epoch):
        
        
        pred_RS = torch.nn.ReLU()(RS_Model(InputTensor_RS))
        
        pred_RA = torch.nn.ReLU()(RA_Model(InputTensor_RA))
        
        pred_Win = torch.nn.ReLU()(Wins_Model(InputTensor_W))
        
        
        loss_RS = criterion_RS(pred_RS, OutputTensor_RS)
        
        loss_RA = criterion_RA(pred_RA, OutputTensor_RA)
        
        loss_Win = criterion_Win(pred_Win, OutputTensor_Win)
        
        
        RS_optimizer.zero_grad()
        
        RA_optimizer.zero_grad()
        
        Win_optimizer.zero_grad()
        
        
        loss_RS.backward()
        
        loss_RA.backward()
        
        loss_Win.backward()
        
        
        RS_optimizer.step()
        
        RA_optimizer.step()
        
        Win_optimizer.step()
        
        
        print('epoch {}, \nloss_RS {}\n'.format(epoch, loss_RS.data.item()))
        
        print('loss_RA {}\n'.format(loss_RA.data.item()))
        
        print('loss_Win {}\n'.format(loss_Win.data.item()))