import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import r2_score,accuracy_score,precision_score,recall_score,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier

st.header('My Own Package of ML')

st.subheader('Load the data')
data=st.file_uploader("Upload Data CSV Or xlsx")

if data is not None :
    if str(data.name).endswith("xls") or str(data.name).endswith("xlsx"):
        data=pd.read_excel(data,encoding='latin-1')
        print(data.head(10))
        st.write(data.head(10))
    elif str(data.name).endswith("csv"):
        data=pd.read_csv(data,encoding='latin-1')
        print(data.head(10))
        st.write(data.head(10))
    else :
            st.info("Please upload a csv or xlsx file")
    	

st.header("Perform EDA")
st.subheader('Clean the Data')
st.text('Clean the Missing Value')

if data is not None :
     data=data.dropna()
     st.write(data.head(10))

st.text('Clean the duplicated Value')     
if data is not None :
	#drop_duplciate_columns
    files=data.drop_duplicates(data)
    st.write(data.head(10))

st.subheader('Visualization')
st.text("Line Plot")
if data is not None :
    numeric_lists=data.select_dtypes(include=['number']).columns
    if len(numeric_lists) != 0:
        colsLine = st.selectbox("Choics column",
                    numeric_lists)
    # for i in numeric_list:
        fig=plt.figure(figsize=(5,5))     
        plt.plot(data[colsLine])
        plt.xlabel(colsLine)
        plt.ylabel("number")
        plt.title(colsLine)
        st.pyplot(fig)
    else :
        st.info("there are no numeric columns")
st.text("Scatter Plot")
if data is not None :
    numeric_list=data.select_dtypes(include=['number']).columns
    if len(numeric_list) != 0:
        colsLine = st.selectbox("Choics Column",
                    numeric_lists)
        column_data = data[colsLine]
        x_values = range(len(column_data))
        fig=plt.figure(figsize=(5,5))
        plt.scatter(x_values, column_data)
        plt.xlabel(colsLine)
        plt.ylabel("Values")
        plt.title(colsLine)
        st.pyplot(fig)
    else :
        st.info("there are no numeric columns")
st.text("heatmap Plot")
if data is not None :
    corr=data.corr()
    if corr.isna().all().all():
        st.info("No correlation")
    else:
        fig=plt.figure(figsize=(5,5))
        sns.heatmap(data.corr())
        st.pyplot(fig)


st.text("pie Plot")
if data is not None :
    numeric_list=data.select_dtypes(include=['number']).columns
    # categorical_list=data.select_dtypes(exclude=['number']).columns
    if len(numeric_list) != 0 :
        colspien= st.selectbox("Choics numerical Column",
                    numeric_lists)
        # colspiec = st.selectbox("Choics categorical Column",
                    # categorical_list)
        fig=plt.figure(figsize=(15,15))
        counts=data[colspien].value_counts()
        print(counts.index)
        plt.pie(counts, labels =counts.index)

        st.pyplot(fig)
    else:
        st.info("Their are no Cataegorical or numerical Columns")



st.header("Deploy Machine Learning")
if data is not None:
    machine_learning_name=['LinearRegression','LogisticRegression','DecisionTreeClassifier',
                           'DecisionTreeRegressor','RandomForestClassifier','RandomForestRegressor','SVC','XGBClassifier','SGDClassifier']
    ML= st.selectbox("Choics Mmachibe Learning Mehtod",
                    machine_learning_name)
    list_of_columns=data.columns

    Target_Column= st.selectbox("Choics the Target column",
                    list_of_columns)
    
    Traing_Column= st.multiselect("Choics the Training columns",
                    list_of_columns)
    is_numeric=data[Target_Column].apply(lambda x: isinstance(x, (int, float))).all()
    if len(Target_Column) != 0  and len(Traing_Column) != 0 and len(list_of_columns) != 0 :
        if not is_numeric:
            pd.get_dummies(data[Target_Column])
            le = LabelEncoder()
            encoded_column = le.fit_transform(data[Target_Column])
            data[Target_Column]=encoded_column
            print(encoded_column)
        x_train,x_test,y_train,y_test=train_test_split(data[Traing_Column],data[Target_Column],test_size=0.2,random_state=42)
        
        if ML=='LinearRegression':
            LR=LinearRegression()
            LR.fit(x_train,y_train)
            pred=LR.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=LR.score(x_train,y_train)
            # resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred)
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            # st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='LogisticRegression':
            LOR=LogisticRegression()
            LOR.fit(x_train,y_train)
            pred=LOR.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=LOR.score(x_train,y_train)
            resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='DecisionTreeClassifier':
            DT=DecisionTreeClassifier()
            DT.fit(x_train,y_train)
            pred=DT.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=DT.score(x_train,y_train)
            resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='DecisionTreeRegressor':
            DTR=DecisionTreeRegressor()
            DTR.fit(x_train,y_train)
            pred=DTR.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=DTR.score(x_train,y_train)
            resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='RandomForestClassifier':
            RFC=RandomForestClassifier()
            RFC.fit(x_train,y_train)
            pred=RFC.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=RFC.score(x_train,y_train)
            resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='RandomForestRegressor':
            RFR=RandomForestRegressor()
            RFR.fit(x_train,y_train)
            pred=RFR.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=RFR.score(x_train,y_train)
            # resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            # st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='SVC':
            SV=SVC()
            SV.fit(x_train,y_train)
            pred=SV.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=SV.score(x_train,y_train)
            # resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            # st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='XGBClassifier':
            XG=XGBClassifier()
            XG.fit(x_train,y_train)
            pred=XG.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=XG.score(x_train,y_train)
            # resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            # st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
        elif ML=='SGDClassifier':
            SG=SGDClassifier()
            SG.fit(x_train,y_train)
            pred=SG.predict(x_test)
            resultr2=r2_score(y_test,pred)
            reslutscore=SG.score(x_train,y_train)
            # resaccuracy=accuracy_score(y_test,pred)
            # resprecision =precision_score(y_test,pred,average='micro')
            # resrecall=recall_score(y_test,pred)
            st.write("the accuracy using r2_score is",resultr2)
            st.write("the accuracy using score is",reslutscore)
            # st.write("the accuracy using accuracy_score is",resaccuracy)
            # st.write("the accuracy using precision_score is",resprecision)
            # st.write("the accuracy using recall_score is",resrecall)
    
    
    else:
        st.info("There are no training columns or target columns")
    

st.text("Line Plot the deifferent between the real value and predicted valure")
if data is not None:
    fig=plt.figure(figsize=(10,10))
    plt.plot(y_test)
    plt.plot(pred)
    st.pyplot(fig)







