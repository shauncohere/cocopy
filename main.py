from logging import PlaceHolder
import streamlit as st
import cohere
import pandas as pd
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Paste your API key here. Remember to not share it publicly 

API_KEY = os.environ['API_KEY']
ENV=os.environ['ENV']

co = cohere.Client(API_KEY)

#gc= pygsheets.authorize(service_file='cohere-sa-ae6a270f57e2.json')

freq=0
n_gens=2
number=0

#sends a single prediction to the cohere generate api
def generate(prompt,model_size="xlarge",n_generations=n_gens, temp=.75, tokens=250, stops=["\n"], freq=freq):
    prediction = co.generate(
                    model=model_size,
                    prompt=prompt,
                    return_likelihoods = 'GENERATION',
                    stop_sequences=stops,
                    max_tokens=tokens,
                    temperature=temp,
                    num_generations=n_generations,
                    k=0,
                    frequency_penalty=freq,
                    p=0.75)
    return(prediction)
#returns max likelihood output
def max_likely(prediction):
    likelihood=[]
    for i, pred in enumerate(prediction.generations):
        likelihood.append(pred.likelihood)
    max_value = np.argmax(likelihood)
    output=prediction.generations[max_value].text
    # if ENV == "dev":  
    #     st.write(likelihood)
    #     st.write(max_value)
    return(output)



#writes output to text file
def write2file(file, model_size, creativity, prompt, summary):
    with open(file,"a") as f:
        f.write("\n**************\n")
        f.write("New Submission\n")
        f.write("**************\n")
        f.write("Model Size:" + model_size + "\n")
        f.write("Creativity:" + str(creativity) + "\n")
        f.write("**************\n")
        f.write(prompt)
        f.write(summary)

def extract_text_starting_with_string(text, string):
    return text.split(string, 1)
#st.image('profile-white-wordmark.png')

def strip_formating_from_string(string):
    string = string.replace("\n", "")
    string = string.replace("\t", "")
    string = string.replace("---", "")
    return string



def print_pdf():
    os.system("out.pdf")

def create_df_with_date_range_as_index(start_date, end_date, columns):
    df = pd.DataFrame(columns=columns)
    df['date'] = pd.date_range(start_date, end_date)
    df = df.set_index('date')
    return df

st.set_page_config(page_title='Ad Copy Generator', initial_sidebar_state = 'auto')
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown(hide_menu_style, unsafe_allow_html=True)


st.title("co:copy ad copy generator")

st.write("This AI powered tool generates ad copy. Give it a try!")
with st.form("my_form"):
    company=st.text_input("What type of company are you?")
    examples=st.text_area("Paste your ad copy examples here seperated by ----")
    nums=st.number_input("How many ad copies would you like to generate?", value=1)
    submitted = st.form_submit_button("Submit")


if submitted:
    for num in range(nums):
        with open('prior_seed.txt') as f:
            prior= f.read()
        prompt= "Here are Google ad headlines for a " + company + "." + examples
        generated=generate(prompt=prompt, model_size="xlarge",tokens=25,temp=.9, stops=["----"])
        generated=max_likely(generated)
        generated=generated.replace("----", "")
        st.write(generated)




