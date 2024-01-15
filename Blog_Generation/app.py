import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import logging

# Add this to your script to configure logging
logging.basicConfig(level=logging.ERROR)

## Function To get response from LLAma 2 model

def getLLamaresponse(input_text,no_words,blog_style):

    ### LLama2 model
    llm=CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML',
                      model_type='llama',
                      config={'max_new_tokens':500,
                              'temperature':0.7})
    
    ## Prompt Template

    template="""
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt=PromptTemplate(input_variables=["blog_style","input_text",'no_words'],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(response)
    return response



st.set_page_config(page_title="Generate Blogs",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text=st.text_input("Enter the Blog Topic")

## creating to more columns for additonal 2 fields
# Create columns
col1, col2 = st.columns([5, 5])

# Column 1: Text Input
with col1:
    options_1 = [100, 200, 500]
    # no_words = st.text_input('No of Words')
    no_words = st.selectbox("Select the number of words",options_1, index=0)
# Column 2: Dropdown Box
with col2:
    # Define options for the selectbox
    options = ('Researchers', 'Data Scientist', 'Common People')
    
    # Use st.selectbox to create the dropdown
    blog_style = st.selectbox('Writing the blog for', options, index=0)
# col1,col2=st.columns([5,5])

# with col1:
#     no_words=st.text_input('No of Words')
# with col2:
#     blog_style=st.selectbox('Writing the blog for',
#                             ('Researchers','Data Scientist','Common People'),index=0)
    
submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))