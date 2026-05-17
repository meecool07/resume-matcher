import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Resume Matcher",
    page_icon="📄",
    layout="wide"
)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\KIIT0001\Desktop\resume_matcher\data\cleaned_resume.csv")
    df['cleaned_resume'] = df['cleaned_resume'].replace([np.nan, None], '').astype(str)
    df = df[df['cleaned_resume'].str.strip() != '']
    return df

# Main app
st.title("📄 AI Resume Matcher")
st.markdown("Match resumes to your job description using TF-IDF & Cosine Similarity")

# Load data
df = load_data()
st.success(f"✅ Loaded {len(df)} resumes successfully!")

st.divider()

# Sidebar filters
st.sidebar.title("⚙️ Filters")

# Category filter in sidebar
all_categories = ["All"] + sorted(list(df['Category'].unique()))
category_filter = st.sidebar.selectbox(
    "Filter by Category:",
    all_categories
)

# Job description input
st.subheader("📝 Enter Job Description")
job_desc = st.text_area(
    "Paste the job description here:",
    height=200,
    placeholder="e.g. Looking for HR Administrator with experience in payroll, recruitment..."
)

# Number of results
top_n = st.slider("Number of top matches to show:", 5, 20, 10)

# Match button
if st.button("🔍 Find Matching Resumes", type="primary"):
    if job_desc.strip() == "":
        st.error("Please enter a job description!")
    else:
        with st.spinner("Matching resumes..."):

            # Apply category filter
            filtered_df = df.copy()
            if category_filter != "All":
                filtered_df = df[df['Category'] == category_filter].copy()
                if len(filtered_df) == 0:
                    st.error(f"No resumes found for category: {category_filter}")
                    st.stop()

            # Clean job desc
            cleaned_job_desc = clean_text(job_desc)

            # Vectorize
            all_texts = list(filtered_df['cleaned_resume']) + [cleaned_job_desc]
            vectorizer = TfidfVectorizer(stop_words=None, max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(all_texts)

            # Cosine similarity
            job_vector = tfidf_matrix[-1]
            resume_vectors = tfidf_matrix[:-1]
            scores = cosine_similarity(job_vector, resume_vectors)[0]

            # Add scores
            filtered_df['match_score'] = scores
            filtered_df['match_score_pct'] = (filtered_df['match_score'] * 100).round(2)

            # Top matches
            top_matches = filtered_df.sort_values(
                'match_score', ascending=False
            ).head(top_n).reset_index(drop=True)
            top_matches.index += 1

        st.divider()

        # Results header
        if category_filter != "All":
            st.subheader(f"🏆 Top {top_n} Matches in '{category_filter}'")
        else:
            st.subheader(f"🏆 Top {top_n} Matching Resumes")

        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Match Score", f"{top_matches['match_score_pct'].iloc[0]}%")
        col2.metric("Best Match Category", top_matches['Category'].iloc[0])
        col3.metric("Average Score", f"{top_matches['match_score_pct'].mean().round(2)}%")

        st.divider()

        # Table
        result_df = top_matches[['ID', 'Category', 'match_score_pct']].rename(
            columns={'match_score_pct': 'Match Score %'}
        )
        st.dataframe(result_df, use_container_width=True)

        # ✅ Download button
        st.download_button(
            label="📥 Download Results as CSV",
            data=result_df.to_csv(index=False),
            file_name="top_matches.csv",
            mime="text/csv"
        )

        st.divider()

        # Bar chart
        st.subheader("📊 Match Score Chart")
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(
            range(len(top_matches)),
            top_matches['match_score_pct'],
            color='steelblue'
        )
        ax.set_xticks(range(len(top_matches)))
        ax.set_xticklabels(top_matches['ID'], rotation=45)
        ax.set_title('Top Resume Match Scores')
        ax.set_ylabel('Match Score %')
        ax.set_xlabel('Resume ID')

        for bar, val in zip(bars, top_matches['match_score_pct']):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{val}%',
                ha='center', va='bottom', fontsize=8
            )

        plt.tight_layout()
        st.pyplot(fig)

        st.divider()

        # Resume previews
        st.subheader("📋 Resume Previews")
        for i, row in top_matches.head(3).iterrows():
            with st.expander(f"#{i} | ID: {row['ID']} | {row['Category']} | Score: {row['match_score_pct']}%"):
                st.write(row['cleaned_resume'][:500] + "...")