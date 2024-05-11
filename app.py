import pickle
import streamlit as st
import pandas as pd
import seaborn as sns

# Load scikit-learn libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv('Medicalpremium.csv')

# Load the model
#@st.cache(allow_output_mutation=True)
@st.cache_resource
def load_model_RF():
    with open('D:\jetbrain\ideProject\PyProject\DACN1\RF_model.pkl', 'rb') as f:
        return pickle.load(f)
def load_model_GBM():
    with open('D:\jetbrain\ideProject\PyProject\DACN1\GBM_model.pkl', 'rb') as f:
        return pickle.load(f)

# Make prediction
def predict(model, data):
    return model.predict(data)

def main():
    st.set_page_config(layout="wide")

    st.sidebar.title("Menu")
    selected = st.sidebar.radio("", ["Home", "Predict", "Contact", "Visualizations"])

    if selected == "Home":
        st.title("PYAR Insurance Company")
        st.header("Mission :")
        st.markdown(
            "**To be the most preferred choice of customers for General Insurance by building Relationships and grow profitably.**")
        st.header("Vision :")
        st.markdown("Leveraging technology to integrate people and processes.")
        st.markdown("To excel in service and performance.")
        st.markdown("To uphold the highest ethical standards in conducting our business.")
        st.header("What is Insurance?")
        st.markdown(
            "Most people have some kind of insurance: for their car, their house, or even their life. Yet most of us don’t stop to think too much about what insurance is or how it works.Put simply, insurance is a contract, represented by a policy, in which a policyholder receives financial protection or reimbursement against losses from an insurance company. The company pools clients’ risks to make payments more affordable for the insured.Insurance policies are used to hedge against the risk of financial losses, both big and small, that may result from damage to the insured or their property, or from liability for damage or injury caused to a third party.")
        st.header("KeyTakeaways")
        st.markdown(
            "Insurance is a contract (policy) in which an insurer indemnifies another against losses from specific contingencies or perils.")
        st.markdown(
            "There are many types of insurance policies. Life, health, homeowners, and auto are the most common forms of insurance")
        st.markdown(
            "The core components that make up most insurance policies are the deductible, policy limit, and premium.")
        st.header("How Insurance Works")
        st.markdown(
            "A multitude of different types of insurance policies is available, and virtually any individual or business can find an insurance company willing to insure them—for a price. The most common types of personal insurance policies are auto, health, homeowners, and life. Most individuals in the United States have at least one of these types of insurance, and car insurance is required by law.Businesses require special types of insurance policies that insure against specific types of risks faced by a particular business. For example, a fast-food restaurant needs a policy that covers damage or injury that occurs as a result of cooking with a deep fryer. An auto dealer is not subject to this type of risk but does require coverage for damage or injury that could occur during test drives.")
        st.subheader("Important Note:")
        st.markdown(
            "To select the best policy for you or your family, it is important to pay attention to the three critical components of most insurance policies: 1.deductible, 2.premium, and 3.policy limit.")

    elif selected == "Predict":
        st.title("Insurance Premium Prediction")
        st.subheader("Enter your information")

        # Load data
        
        df = load_data()
        

        # Load model
        model_RF = load_model_RF()
        model_GBM = load_model_GBM()
        
        

        # Collect user inputs
        Age = st.number_input("Age", min_value=1, max_value=100)
        Height = st.slider('Height(cm)', 10, 200)
        Weight = st.slider('Weight(kg)', 10, 150)
        Diabetes = st.checkbox('Diabetes')
        BP = st.checkbox('Blood pressure')
        Transplants = st.checkbox('Transplants')
        ChronicDiseases = st.checkbox('Chronic Diseases')
        KnownAllergies = st.checkbox('Known Allergies')
        HistoryOfCancerInFamily = st.checkbox('History Of Cancer In Family')
        NumberOfMajorSurgeries = st.number_input("Number Of Major Surgeries", min_value=0, max_value=10)

        # Convert checkbox values to binary
        Diabetes = 1 if Diabetes else 0
        BP = 1 if BP else 0
        Transplants = 1 if Transplants else 0
        ChronicDiseases = 1 if ChronicDiseases else 0
        KnownAllergies = 1 if KnownAllergies else 0
        HistoryOfCancerInFamily = 1 if HistoryOfCancerInFamily else 0
        BMI = Weight / ((Height / 100) ** 2)

        # Make prediction
        prediction = predict(model_RF, [[Age, Diabetes, BP, Transplants, ChronicDiseases, BMI, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries]])
        
        st.success(f"Hey! By RF model Your health insurance premium price is Rs. {prediction[0]:.2f}")
        
        prediction = predict(model_GBM, [[Age, Diabetes, BP, Transplants, ChronicDiseases, BMI, KnownAllergies, HistoryOfCancerInFamily, NumberOfMajorSurgeries]])
        
        st.success(f"Hey! By GBM model Your health insurance premium price is Rs. {prediction[0]:.2f}")



    elif selected == "Contact":
        st.title("PYAR Insurance Company")
        st.subheader("Reach us at:")
        st.write("abc: 1111111111")
        st.write("def: 2222222222")
        st.write("pqr: 3333333333")
        st.write("lmn: 4444444444")
        st.write("pyarinsurance@pyar.com")
        st.write("insurance@pyar.com")

    elif selected == "Visualizations":
        st.title("Visualizations of the Data")
        st.subheader("DataFrame")
        df = load_data()
        st.write(df)
        
        st.write(df.describe().T)

        # Feature Importance
        X = df.drop('PremiumPrice', axis=1)
        y = df['PremiumPrice']
        model = RandomForestClassifier()
        model.fit(X, y)
        feat_imp = pd.Series(model.feature_importances_, index=X.columns)
        st.subheader("Feature Importance")
        st.bar_chart(feat_imp.nlargest(10))

        pr_lab = ['Low', 'Basic', 'Average', 'High', 'SuperHigh']
        df['PremiumLabel'] = pd.cut(df['PremiumPrice'], bins=5, labels=pr_lab, precision=0)
        

        # Confusion Matrix
        scaler = StandardScaler()
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=43)
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestClassifier()
        rf.fit(X_train, Y_train)
        y_pred_rf = rf.predict(X_test)

        cm_rf = confusion_matrix(Y_test, y_pred_rf)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm_rf, annot=True, fmt="d", ax=ax)
        st.pyplot(fig)
        fig2 = sns.displot(x='Age', data=df, aspect=10 / 6, kde=True)
        
        st.pyplot(fig2)

        fig3, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x=df.Age, y=df.PremiumPrice).set_title('Insurance Premium Price by Age')
        st.pyplot(fig3)

        fig4, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(x='PremiumLabel', data=df, ax=ax).set_title('Distribution of the Insurance Premium Price')
        st.pyplot(fig4)

        # loi khong hien thi labdel 
        #st.write(df.groupby(['PremiumLabel', 'NumberOfMajorSurgeries'])['NumberOfMajorSurgeries'].count())

        fig5, ax = plt.subplots()
        sns.barplot(data=df, x="Diabetes", y="PremiumPrice", ax=ax)
        plt.title('Insurance Premium Price for Diabetic vs Non-Diabetic Patients')
        st.pyplot(fig5)

        plt.figure(figsize=(10, 6))
        sns.displot(df, x="PremiumPrice", hue="Diabetes", kind="kde", fill=True)
        plt.title('Density plot for Diabetic vs Non-Diabetic Patients', fontsize=12, fontweight='bold')
        st.pyplot(plt.gcf())

        fig6, ax = plt.subplots()
        sns.barplot(data=df, x="AnyTransplants", y="PremiumPrice", ax=ax)
        plt.title('Insurance Premium Price for Patients with/without Any Transplants')
        st.pyplot(fig6)
        
        #lam toi cell 236
if __name__ == '__main__':
    main()
