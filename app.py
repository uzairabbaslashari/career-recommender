
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("dataset.csv")

# Vectorization
tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(df["skills"])

# Roadmaps
roadmaps = {
    "Data Scientist": ["Learn Python", "Learn Statistics", "Machine Learning", "Projects"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React"],
    "Graphic Designer": ["Design Basics", "Photoshop", "Illustrator"],
    "Doctor": ["MBBS", "Specialization"],
    "Teacher": ["Bachelor Degree", "Teaching Skills"],
    "Cyber Security Analyst": ["Networking", "Security Basics", "Ethical Hacking"],
    "Mobile App Developer": ["Flutter/Android", "UI Design", "App Projects"],
    "AI Engineer": ["Python", "Deep Learning", "AI Projects"],
    "Digital Marketer": ["SEO", "Social Media", "Marketing Strategy"]
}

# Recommendation function
def recommend_careers(user_input):
    user_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(user_vec, vectors)
    
    scores = list(enumerate(similarity[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    results = []
    for i in scores[:3]:
        results.append(df.iloc[i[0]]["career"])
    
    return results

# UI
st.title("🎓 Career Path Recommender")

user_input = st.text_input("Enter your skills/interests:")

if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("Please enter something!")
    else:
        results = recommend_careers(user_input)
        
        for career in results:
            st.subheader(career)
            roadmap = roadmaps.get(career, [])
            
            for step in roadmap:
                st.write("➡️", step)
