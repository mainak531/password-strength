import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_URL = "http://13.48.1.212:8000"  # Change this to your deployed API URL
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

st.set_page_config(
    page_title="🔐 Password Strength Classifier",
    page_icon="🔐",
    layout="wide"
)

# Check API health on startup
@st.cache_data(ttl=60)
def check_api_health():
    try:
        response = requests.get(HEALTH_ENDPOINT)
        return response.status_code == 200
    except:
        return False

API_HEALTHY = check_api_health()

# Sidebar
st.sidebar.title("🔐 Password Strength Classifier")
st.sidebar.markdown("---")

# API Status
if API_HEALTHY:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Unavailable - Make sure FastAPI server is running")
    st.sidebar.info("Run: uvicorn app:app --reload --host 0.0.0.0 --port 8000")

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Single Password", "📊 Batch Analysis", "ℹ️ About"]
)

# Password strength mapping
strength_colors = {
    'weak': 'red',
    'medium': 'orange',
    'strong': 'green'
}

def check_password_strength(password):
    """Call the ML API to check password strength"""
    if not API_HEALTHY:
        st.error("❌ API not available. Please check if the FastAPI server is running.")
        return None
    
    try:
        with st.spinner("🤖 Analyzing with ML model..."):
            response = requests.post(
                PREDICT_ENDPOINT,
                json={"password": password}
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            if response.status_code == 503:
                st.error("Model not loaded on server")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the FastAPI server. Make sure it's running on port 8000")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Page 1: Single Password Checker
if page == "🔍 Single Password":
    st.title("🔐 Password Strength Checker")
    st.markdown("Enter a password to check its strength using our **94% accurate ML model**")
    
    if not API_HEALTHY:
        st.warning("⚠️ API is not connected. Please start the FastAPI server first.")
        st.code("uvicorn app:app --reload --host 0.0.0.0 --port 8000")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password here...",
            key="single_pwd"
        )
        
        col_check, col_clear = st.columns([1, 5])
        with col_check:
            check_button = st.button("🔍 Check Strength", type="primary", use_container_width=True)
        
        if check_button and password:
            result = check_password_strength(password)
            
            if result:
                strength = result['strength']
                confidence = result['confidence']
                prob = result['class_probabilities']
                features = result['features']
                
                st.markdown("---")
                
                # Main result
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    color = strength_colors[strength]
                    st.markdown(f"### Strength: **:{color}[{strength.upper()}]**")
                    st.progress(confidence, text=f"ML Confidence: {confidence:.1%}")
                
                # Class probabilities
                st.subheader("📊 Class Probabilities")
                prob_df = pd.DataFrame({
                    'Class': list(prob.keys()),
                    'Probability': list(prob.values())
                })
                fig = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    color='Class',
                    color_discrete_map=strength_colors,
                    title="ML Model Prediction Probabilities"
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature analysis
                st.subheader("🔍 Password Features")
                col_x, col_y, col_z = st.columns(3)
                
                with col_x:
                    st.metric("Length", features['length'])
                    st.metric("Lowercase Frequency", f"{features['lowercase_freq']:.2f}")
                
                with col_y:
                    st.metric("Has Uppercase", "✅" if features['has_uppercase'] else "❌")
                    st.metric("Has Digit", "✅" if features['has_digit'] else "❌")
                
                with col_z:
                    st.metric("Has Special", "✅" if features['has_special'] else "❌")
        
        elif check_button and not password:
            st.warning("Please enter a password")

    with col2:
        st.markdown("### 📝 Password Tips")
        st.info(
            """
            **Strong passwords have:**
            - ✅ 12+ characters
            - ✅ Uppercase & lowercase
            - ✅ Numbers & special chars
            - ❌ No dictionary words
            - ❌ No personal info
            """
        )
        
        st.markdown("### 🎯 Try Examples")
        if st.button("📌 Try 'password123'"):
            st.session_state['single_pwd'] = "password123"
        if st.button("📌 Try 'P@ssw0rd!'"):
            st.session_state['single_pwd'] = "P@ssw0rd!"
        if st.button("📌 Try 'MyStr0ngP@ss2024'"):
            st.session_state['single_pwd'] = "MyStr0ngP@ss2024"

# Page 2: Batch Analysis
elif page == "📊 Batch Analysis":
    st.title("📊 Batch Password Analysis")
    st.markdown("Analyze multiple passwords at once using our ML model")
    
    if not API_HEALTHY:
        st.warning("⚠️ API is not connected. Please start the FastAPI server first.")
        st.stop()
    
    tab1, tab2 = st.tabs(["📁 Upload File", "✏️ Manual Entry"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload a CSV or TXT file with passwords",
            type=['csv', 'txt'],
            help="CSV should have a 'password' column. TXT should have one password per line."
        )
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'password' in df.columns:
                    passwords = df['password'].tolist()
                else:
                    st.error("CSV must contain a 'password' column")
                    st.stop()
            else:  # txt file
                content = uploaded_file.read().decode('utf-8')
                passwords = [p.strip() for p in content.split('\n') if p.strip()]
            
            st.info(f"📊 Found {len(passwords)} passwords to analyze")
            
            if st.button("🚀 Start Analysis", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                for i, pwd in enumerate(passwords):
                    status_text.text(f"Analyzing {i+1}/{len(passwords)}: {pwd[:20]}...")
                    result = check_password_strength(pwd)
                    if result:
                        results.append({
                            'password': pwd[:30] + ('...' if len(pwd) > 30 else ''),
                            'strength': result['strength'],
                            'confidence': f"{result['confidence']:.2%}",
                            'length': result['features']['length']
                        })
                    progress_bar.progress((i + 1) / len(passwords))
                
                status_text.text("✅ Analysis complete!")
                
                if results:
                    results_df = pd.DataFrame(results)
                    
                    # Summary stats
                    st.subheader("📈 Summary Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", len(results_df))
                    with col2:
                        weak_count = len(results_df[results_df['strength'] == 'weak'])
                        st.metric("Weak", weak_count, delta=f"{weak_count/len(results_df)*100:.1f}%")
                    with col3:
                        medium_count = len(results_df[results_df['strength'] == 'medium'])
                        st.metric("Medium", medium_count)
                    with col4:
                        strong_count = len(results_df[results_df['strength'] == 'strong'])
                        st.metric("Strong", strong_count)
                    
                    # Distribution chart
                    fig = px.pie(
                        results_df,
                        names='strength',
                        title='Password Strength Distribution',
                        color='strength',
                        color_discrete_map=strength_colors
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        csv,
                        "password_analysis_results.csv",
                        "text/csv"
                    )
    
    with tab2:
        st.subheader("Enter passwords manually")
        passwords_text = st.text_area(
            "Enter one password per line",
            height=200,
            placeholder="password123\nP@ssw0rd!\nMyStr0ngP@ss"
        )
        
        if st.button("Analyze Manual Entries", type="primary"):
            if passwords_text:
                passwords = [p.strip() for p in passwords_text.split('\n') if p.strip()]
                
                results = []
                for pwd in passwords:
                    result = check_password_strength(pwd)
                    if result:
                        results.append({
                            'password': pwd[:30] + ('...' if len(pwd) > 30 else ''),
                            'strength': result['strength'],
                            'confidence': f"{result['confidence']:.2%}",
                            'length': result['features']['length']
                        })
                
                if results:
                    results_df = pd.DataFrame(results)
                    st.subheader("📋 Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "manual_analysis.csv", "text/csv")

# Page 3: About
else:
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🔐 ML-Powered Password Strength Classifier
    
    This application uses a **Machine Learning model with 94.2% accuracy** to classify password strength 
    into three categories: **Weak**, **Medium**, or **Strong**.
    
    ### 🎯 Features
    - **Real-time password strength checking** with ML model
    - **Batch analysis** for multiple passwords
    - **Interactive visualizations** of predictions
    - **Confidence scores** for each prediction
    
    ### 🧠 Model Details
    - **Algorithm**: XGBoost
    - **Accuracy**: 94.2%
    - **Features**: 
      - Character-level TF-IDF (99 features)
      - Password length
      - Lowercase character frequency
    
    ### 📊 Model Performance
    """)
    
    # Model performance metrics
    metrics = pd.DataFrame({
        'Class': ['Weak', 'Medium', 'Strong'],
        'Precision': [0.91, 0.94, 0.95],
        'Recall': [0.74, 0.98, 0.92],
        'F1-Score': [0.82, 0.96, 0.93]
    })
    
    st.dataframe(metrics, use_container_width=True)
    
    st.markdown("""
    ### 🚀 Technology Stack
    - **Backend**: FastAPI
    - **ML**: XGBoost, scikit-learn
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    
    ### 📁 Project Structure
    ```
    ├── app.py                 # FastAPI ML service
    ├── frontend.py             # Streamlit UI
    ├── password_strength_*.pkl # Trained XGBoost model
    └── requirements.txt        # Dependencies
    ```
    """)
    
    st.info("⭐ This project achieved 94.2% accuracy using XGBoost with character-level features!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using FastAPI + Streamlit")
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Check API Status"):
    st.cache_data.clear()
    st.rerun()
