import argparse,os
import sys
import math
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def define_args(arg_parser):

    arg_parser.add_argument('--feature_folder', nargs='*', default=['features_selected'], help='Gringo input files')
    arg_parser.add_argument('--performance_folder', nargs='*', default=['performance_selected'], help='Gringo input files')
    arg_parser.add_argument('--cutoff', nargs='*', default=['200'], help='Gringo input files')
    arg_parser.add_argument('--ml_models_folder', nargs='*', default=['ml_models'], help='Gringo input files') 
    arg_parser.add_argument('--ml_hyper_folder', nargs='*', default=['ml_hyper'], help='Gringo input files') 

def checkMakeFolder(fdname):
    if not os.path.exists(fdname):
        os.makedirs(fdname)

def check_content(fdname):
    if os.listdir(fdname) == []:
        return False
    else:
        return True

def cleanFolder(fdnames):   
    ans=input('Models existed. Need to retrain models? y/n')
    if ans =='y':
        for file_in in fdnames:
            if os.path.exists(file_in):
                os.system('rm -r '+file_in+'/*')

#write to evaluation file2 
#evaluation/result2.csv
def write2eva2(algname,slv,time):
    fname='evaluation/result2.csv'
    with open (fname,'a') as f:
        cont=','.join([algname,str(slv),str(time)])
        f.write(cont+'\n')
#the objective function to minimize, tuning hyperparameters
#relative_score
#max_relative_score
#min_relative_score
#neg_mean_squared_error
def relative_score(y_true, y_pred):
		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -sum(res)/float(len(res))

def max_relative_score(y_true, y_pred):

		res=[]
		for i in range(len(y_true)):
			if y_true[i]>y_pred[i]:
				res.append((y_true[i]-y_pred[i])/(y_true[i]))
			else:
				res.append((y_pred[i]-y_true[i])/(y_true[i]))
		return -max(res)

#print solved percentage and avg solving only time
def printSvdPercAvgTime(p,runtime,maxtime,printresult=True):
	#success
	sucs=[]
	for i in runtime:
		if i<maxtime-1:
			sucs.append(i)
	if len(sucs)!=0:
		if printresult:
			print(p,float(len(sucs))/len(runtime),"/",float(sum(sucs))/len(sucs))
		return float(len(sucs))/len(runtime), float(sum(sucs))/len(sucs)
	else:
		if printresult:
			print(p,float(0),"/",float(0))
		return 0,0

#print solved percentage and real avg runtime
def printSvdPercAvgTime2(p,runtime,maxtime,printresult=True):
	#success
	sucs=[]
	time_real=[]
	for i in runtime:
		time_real.append(i)
		if i<maxtime-1:
			sucs.append(i)
	if len(sucs)!=0:
		if printresult:
			print(p,float(len(sucs))/len(runtime),"/",float(sum(time_real))/len(runtime))
		return float(len(sucs))/len(runtime), float(sum(time_real))/len(runtime)
	else:
		if printresult:
			print(p,float(0),"/",float(maxtime))
		return 0,0

#split 80% trainset into validSet, trainSet with specified binNum and which bin.
#bin=0, binNum=5.
#the last bin for validing, first 4bins for training.
def splitTrainValid(datasetX,bin,binNum):
	bin_size=int(math.ceil(len(datasetX)/binNum))
	if bin==0:
		return np.array(datasetX[bin_size:]),np.array(datasetX[:bin_size])
	elif bin==binNum-1:
		return np.array(datasetX[:(binNum-1)*bin_size]),np.array(datasetX[-bin_size:])
	else:
		return np.append(datasetX[:bin_size*(bin)],datasetX[bin_size*(bin+1):],axis=0),np.array(datasetX[bin_size*(bin):bin_size*(bin+1)])



def drawLine():
    print("------------------------------------------------")


def getfromindex(input_df,csvfile):
    df=pd.read_csv(csvfile)
    #print(input_df)
    #print(df)
    instance_value=df['Instance_index'].values
    #print(instance_value)
    return input_df.loc[instance_value]

def machine_learning(args,ml_group,ml_last_group):

    feature_folder=args.feature_folder[0]+'/'+ml_group
    performance_folder=args.performance_folder[0]+'/'+ml_group
    cutoff=args.cutoff[0]

    ml_outfolder=args.ml_models_folder[0]+'/'+ml_group
    ml_hyperfolder=args.ml_hyper_folder[0]+'/'+ml_group


    ml_last_outfolder=args.ml_models_folder[0]+'/'+ml_last_group
    ml_hyperfolder=args.ml_hyper_folder[0]+'/'+ml_group

    checkMakeFolder(ml_hyperfolder)
    checkMakeFolder(ml_outfolder)

    #set according to your cutoff time
    TIME_MAX=int(cutoff)
    #use varing PENALTY policy or 1000s fixed
    VARING_PENALTY=False

    #set PENALTY_TIME, we can set as 200, PAR10, or PARX
    PENALTY_TIME=int(cutoff)



    score_functions=[make_scorer(relative_score),make_scorer(max_relative_score),"neg_mean_squared_error"]
    # here choose "neg_mean_squared_error"
    score_f=score_functions[2]


    featureFile=feature_folder+'/'+os.listdir(feature_folder)[0]
    featureValue=pd.read_csv(featureFile)
    featureValue=featureValue.set_index(featureValue.columns[0])
    allCombine=featureValue.copy()

    performanceFile=performance_folder+'/'+os.listdir(performance_folder)[0]
    performanceValue=pd.read_csv(performanceFile)
    performanceValue=performanceValue.set_index(performanceValue.columns[0])
    algorithmNames=performanceValue.columns.values
    performanceValue.columns=["runtime_"+algo for algo in algorithmNames]
    allCombine=allCombine.join(performanceValue)



    #remove duplicated
    allCombine = allCombine[~allCombine.index.duplicated(keep='first')]
    allCombine.sort_index()


    featureList=allCombine.columns.values[:-len(algorithmNames)]
    print("[Feature used]:",featureList)


    #drop "na" rows
    allCombine=allCombine.dropna(axis=0, how='any')

    #drop "?" rows
    for feature in featureList[1:]:
        if allCombine[feature].dtypes=="object":
            # delete from the pd1 rows that contain "?"
            allCombine=allCombine[allCombine[feature].astype("str")!="?"]


    algs=["runtime_"+algo for algo in algorithmNames]
    allRuntime=allCombine[algs]
    print(allRuntime.shape,allRuntime)
    oracle_value=np.amin(allRuntime.values, axis=1)
    oracle_index=np.argmin(allRuntime.values, axis=1)
    Oracle_name=[algorithmNames[oracle_index[i]] for i in range(len(oracle_index))]
    allCombine["Oracle_value"]=oracle_value
    allCombine["Oracle_name"]=Oracle_name
    allCombine["Instance_index"]=allCombine.index.values
    
    if ml_group==ml_last_group:

        np.random.seed(123)
        random.seed(123)
        #shuffle
        #print(np.random.permutation(len(allCombine)))
        allCombine=allCombine.iloc[np.random.permutation(len(allCombine))]

        #print('all',len(allCombine))
        # get leave out data 15% of the full data:
        leaveIndex=random.sample(range(allCombine.shape[0]), int(allCombine.shape[0]*0.15))
        #print('leaveIndex',len(leaveIndex))
        mlIndex=list(range(allCombine.shape[0]))
        for i in leaveIndex:
            if i in mlIndex:
                mlIndex.remove(i)

        leaveSet=allCombine.iloc[leaveIndex]
        mlSet=allCombine.iloc[mlIndex]
        #print('mlSet',len(mlSet))
        # get testing data 20% of the full data:
        testIndex=random.sample(range(mlSet.shape[0]), int(mlSet.shape[0]*0.2))
        #print('testIndex',len(testIndex))
        trainIndex=list(range(mlSet.shape[0]))
        for i in testIndex:
            if i in trainIndex:
                trainIndex.remove(i)
        #print('trainIndex',len(testIndex))
        testSet=mlSet.iloc[testIndex]
        trainSetAll=mlSet.iloc[trainIndex]
        trainSetAll.to_csv(ml_outfolder+'/trainSetAll.csv')
        trainSet,validSet=splitTrainValid(trainSetAll,0,5)

        trainSet=pd.DataFrame(trainSet,columns=trainSetAll.columns)
        validSet=pd.DataFrame(validSet,columns=trainSetAll.columns)

    
        print("ALL after preprocess:",len(mlSet))
        print("trainAll:",len(trainSetAll))
        print("--trainSet:",trainSet.shape)
        print("--validSet:",validSet.shape)
        print("Validation set:",testSet.shape)
        print("Test set:",leaveSet.shape)

        trainSet.to_csv(ml_outfolder+"/trainSet.csv")
        validSet.to_csv(ml_outfolder+"/validSet.csv")
        testSet.to_csv(ml_outfolder+"/testSet.csv")
        leaveSet.to_csv(ml_outfolder+"/leaveSet.csv")

    else:# index are same as last group
        trainSetAll=getfromindex(allCombine,ml_last_outfolder+"/trainSetAll.csv")
        trainSet=getfromindex(allCombine,ml_last_outfolder+"/trainSet.csv")
        validSet=getfromindex(allCombine,ml_last_outfolder+"/validSet.csv")
        testSet=getfromindex(allCombine,ml_last_outfolder+"/testSet.csv")
        leaveSet=getfromindex(allCombine,ml_last_outfolder+"/leaveSet.csv")

        print("ALL after preprocess:",len(trainSet)+len(validSet)+len(testSet)+len(leaveSet))
        print("trainAll:",trainSetAll.shape)
        print("--trainSet:",trainSet.shape)
        print("--validSet:",validSet.shape)
        print("Validation set:",testSet.shape)
        print("Test set:",leaveSet.shape)       
         
        trainSetAll.to_csv(ml_outfolder+'/trainSetAll.csv')
        trainSet.to_csv(ml_outfolder+"/trainSet.csv")
        validSet.to_csv(ml_outfolder+"/validSet.csv")
        testSet.to_csv(ml_outfolder+"/testSet.csv")
        leaveSet.to_csv(ml_outfolder+"/leaveSet.csv")


if __name__ == "__main__":
    print('\nMachine learning model building...')
    parser = argparse.ArgumentParser()
    define_args(parser)
    args = parser.parse_args()


    ml_outfolder=args.ml_models_folder[0]
    ml_hyperfolder=args.ml_hyper_folder[0]

    checkMakeFolder(ml_hyperfolder)
    checkMakeFolder(ml_outfolder)

    if check_content(ml_hyperfolder) or check_content(ml_hyperfolder):
        cleanFolder([ml_hyperfolder,ml_outfolder])

    #evaluating
    if not os.path.exists('evaluation'):
        os.system('mkdir evaluation')
    os.system('rm evaluation/*')
    with open('evaluation/result.csv','w') as f:
        f.write('method,solving,time\n')    

    with open('evaluation/result2.csv','w') as f:
        f.write('test\n')

    feature_folder=args.feature_folder[0]
    feature_groups=os.listdir(feature_folder)
    for group_index,ml_group in enumerate(feature_groups[::-1]):
        machine_learning(args,ml_group,feature_groups[-1])
    
