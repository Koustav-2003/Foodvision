import streamlit as st
import requests

st.set_page_config(
    
    page_title='Foodvision',
    page_icon='🍕',
    layout='centered'
)

##Connect to the FastAPI backend
API_URL="http://127.0.0.1:8000/predict"

st.time_input('🍽️ FoodVision')

st.write('Please upload a food image and find out what it is')

uploaded_file=st.file_uploader(
    'Choose an Food Image',
    type=['jpeg','jpg','png','webp']
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption='Uploaded image',
        use_container_width=True
    )
if st.button('Predict food'):
    try:
        with st.spinner('Classifying your Image...'):
            response=requests.post(
                API_URL,
                files={
                    'file':(
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                },
                timeout=120
            )
            
            response.raise_for_status()
            result=response.json()
            predictions=result['predictions']
            
            st.success(
                f'Predicted Food:{
                    predictions[0]['class'].replace('_',' ').title()}'
            )
            st.metric(
                label='Confidence',
                value=f'{predictions[0]['confidence']}%'
            )
            
            #Display all top 5 predictions
            
            for prediction in predictions:
                food_name=prediction['class'].replace('_',' ').title()
                confidence=prediction['confidence']
                
                st.write(f'**{food_name}** -{confidence}%')
                st.progress(confidence/100)
    
    except requests.exceptions.ConnectionError:
        st.error(
            'Could not connect to Backend.Please make Sure FastAPI is running'
        )  
    
    except requests.exceptions.RequestException as e:
        st.error(f'API request failed:{e}')
        
    except (KeyError,ValueError) as e:
        st.error(f'Invalid API response :{e}')                 