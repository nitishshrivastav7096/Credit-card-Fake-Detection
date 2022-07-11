import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def welcome():
    print("Welcome in Fake Credit Card Predication System")
    print("Press ENTER key to Proceed")
    input()
    
def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content=os.listdir(cur_dir)
    for file_name in content:
        if file_name.split('.')[-1]=='csv':
            csv_files.append(file_name)
    return csv_files
            

def select_csv_file(csv_files):
    i=0
    for file_name in csv_files:
        print(i,'------->',file_name)
        i+=1
    return csv_files[int(input("Select Your csv file"))]     



def main():
    welcome()
    try:
        csv_files=checkcsv()
        csv_file=select_csv_file(csv_files)
        print("Reading the CSV file.........")

        #Loading the dataset to a pandas DataFrame
        dataset=pd.read_csv(csv_file)
        
        print("Creating Dataset..........")
        #First 5 rows of the dataset
        print(dataset.head())
      

        #Last 5 rows of the dataset
        print(dataset.tail())
       

        #dataset information 
        print(dataset.info())

        #checking the number of missing values in each column
        print(dataset.isnull().sum())

        #distribution of legit transactions & fraudulent transactions
        dataset['Class'].value_counts()

        #This dataset is highly unblanced
        #0-->Normal Transition
        #1-->Fraudulent Transition

        #separating the data for analysis
        legit=dataset[dataset.Class==0]
        fraud=dataset[dataset.Class==1]

        print(legit.shape,fraud.shape)

        #statistical measure of the data
        print(legit.Amount.describe())
        print(fraud.Amount.describe())

        #compare the values for both of transition
        dataset.groupby('Class').mean()

        #Under - sampling ( manage the unblance data)
        #Build a sample dataset containing similar distribution of normal transations and fraudulent transactions

        legit_sample=legit.sample(n=492)

        #Concatenation two dataset
        new_dataset=pd.concat([legit_sample,fraud],axis=0)
        print(new_dataset['Class'].value_counts())
        print(new_dataset.groupby('Class').mean())

        #Splitting the data into Features and Targets
        x=new_dataset.drop(columns='Class',axis=1)   #axis =1 represet the column)
        y=new_dataset['Class']
        print("Dataset is Created")

        #Split the data into Training data & Test data
        print("Creating Machine Learning Model.......")
        s=float(input("Enter a test size(between 0 and 1)"))
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=s,stratify=y,random_state=2)


        #Model Training
        model=LogisticRegression()

        #Training the Logistic Regression Model with Training Data
        model.fit(x_train,y_train)

        #Evalution
        training_predication=model.predict(x_train)
        training_predication_accuracy=accuracy_score(training_predication,y_train)
        print("The Training data accuracy is %2.2f%%"%(training_predication_accuracy*100))

        y_predicat=model.predict(x_test)
        accuracy=accuracy_score(y_predicat,y_test)
        print("Press ENTER key to know Test data accuracy")
        input()
        print("The Training data accuracy is %2.2f%%"%(accuracy*100))

        #Model ready to use by user
        
        print("Machine Learning Model is created Now you can use this Model")
        print("Press ENTER key to use this Model")
        input()
        print("ENTER a data")
        user_data=list(map(float,input().split()))
        user_data_numpy=np.asarray(user_data)
        new_user_data=user_data_numpy.reshape(1,-1)

        pred=model.predict(new_user_data)
        if(pred==0):
            print("Transition is Real")
        else:
            print("Transition is Fraud")
        

    
        
        
    except FileNotFoundError:
       print("CSV file is detected")
       print("Press ENTER to exit")
       input()
       exit()
       



if __name__=="__main__":
    main()
    input()
