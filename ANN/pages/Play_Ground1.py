import streamlit as st

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",  # Add your desired icon symbol here
)

st.title("Main page")
st.sidebar.success("Select a page above.")

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
num_sam = st.sidebar.slider(min_value=1,max_value=10000,label='Number of samples')
random = st.sidebar.slider(min_value=1,max_value=100,label='Random state')
Batch = st.sidebar.slider(f"Batch", min_value=1, step=1,max_value=1000)
epoch = st.sidebar.number_input(f"epochs", min_value=1, step=1,max_value=1000)

## Finding Loss Function
import tensorflow as tf
import keras 
from keras.models import Sequential
from keras.layers import Dense,InputLayer

model = Sequential()
model.add(InputLayer(input_shape=(2,)))
Hidden_layer=st.sidebar.number_input("Hidden Layers", min_value=1, step=1)
hidden_layers_config=[]
for layer in range(Hidden_layer):
    neurons = st.sidebar.number_input(f"Number of Neurons in Layer {layer}", min_value=1, step=1)
    activation = st.sidebar.selectbox(f"Activation Function for Layer {layer}", ["relu", "sigmoid", "tanh","softmax"])
    hidden_layers_config.append((neurons, activation))
for neurons, activation in hidden_layers_config:
    model.add(Dense(neurons, activation=activation, use_bias=True)) 


if st.sidebar.button("Submit"):

    fv,cv = make_classification(n_samples=num_sam,n_features=2, n_informative=2,n_redundant=0,n_classes=2,random_state=random)
    df = pd.DataFrame(fv,columns=['Feature_1', 'Feature_2'])
    df['label']=cv


    ## Shows the Scatter plot
    import seaborn as sns
    scat = sns.scatterplot(data=df,x='Feature_1',y='Feature_2',hue='label')
    st.pyplot(scat.figure)

    
    model.summary()
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])

    y=df.pop('label')
    X=df

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,
                                                    random_state = 12, stratify = cv)

    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    model.fit(x_train,y_train,epochs=epoch,validation_split=0.3)
    history=model.fit(x_train,y_train,epochs=epoch,batch_size=Batch,validation_split=0.3)


    import matplotlib.pyplot as plt
    # Create plot

    fig,ax = plt.subplots()
    plt.plot(range(1, epoch+1), history.history['loss'], label='Train')
    plt.plot(range(1, epoch+1), history.history['val_loss'], label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    st.pyplot(fig)

    import mlxtend
    from mlxtend.plotting import plot_decision_regions

    # Plot decision regions
    fig, ax = plt.subplots()
    plot_decision_regions(x_test, y_test.values, clf=model, ax=ax)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Decision Regions')
    st.pyplot(fig)