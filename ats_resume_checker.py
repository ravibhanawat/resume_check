import streamlit as st
import PyPDF2
import docx
import re
import nltk
import spacy
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Please install the spaCy English model: python -m spacy download en_core_web_sm")
        return None

class ATSResumeChecker:
    def __init__(self):
        self.nlp = load_spacy_model()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # ATS-friendly keywords by category
        self.skill_keywords = {
            'technical': ['python', 'java', 'javascript', 'sql', 'html', 'css', 'react', 'angular', 
                         'node.js', 'aws', 'azure', 'docker', 'kubernetes', 'git', 'api', 'rest',
                         'microservices', 'agile', 'scrum', 'devops', 'ci/cd', 'tensorflow', 'pytorch',
                         'machine learning', 'data science', 'artificial intelligence'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical',
                           'creative', 'adaptable', 'collaborative', 'innovative', 'strategic'],
            'action_verbs': ['achieved', 'developed', 'implemented', 'managed', 'led', 'created',
                           'improved', 'increased', 'reduced', 'optimized', 'designed', 'built',
                           'launched', 'delivered', 'coordinated', 'supervised']
        }
        
        # ATS scoring criteria
        self.scoring_criteria = {
            'contact_info': 10,
            'keywords_match': 25,
            'action_verbs': 15,
            'formatting': 15,
            'length': 10,
            'sections': 15,
            'readability': 10
        }

    def extract_text_from_pdf(self, uploaded_file):
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return None

    def extract_text_from_docx(self, uploaded_file):
        try:
            doc = docx.Document(uploaded_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_contact_info(self, text):
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        return {
            'emails': emails,
            'phones': phones,
            'has_contact': len(emails) > 0 and len(phones) > 0
        }

    def analyze_keywords(self, text, job_description=""):
        text_processed = self.preprocess_text(text)
        words = text_processed.split()
        
        # Count skill keywords
        found_keywords = {'technical': [], 'soft_skills': [], 'action_verbs': []}
        
        for category, keywords in self.skill_keywords.items():
            for keyword in keywords:
                if keyword in text_processed:
                    found_keywords[category].append(keyword)
        
        # Calculate keyword density
        total_words = len(words)
        keyword_density = sum(len(cat_keywords) for cat_keywords in found_keywords.values()) / total_words * 100
        
        return found_keywords, keyword_density

    def check_formatting(self, text):
        lines = text.split('\n')
        
        # Check for common resume sections
        sections = ['experience', 'education', 'skills', 'summary', 'objective', 'projects']
        found_sections = []
        
        for line in lines:
            line_lower = line.lower().strip()
            for section in sections:
                if section in line_lower and len(line_lower) < 50:
                    found_sections.append(section)
        
        # Check for bullet points
        bullet_count = sum(1 for line in lines if line.strip().startswith(('•', '◦', '-', '*')))
        
        # Check for inconsistent formatting
        has_bullets = bullet_count > 0
        has_sections = len(set(found_sections)) >= 3
        
        return {
            'sections_found': list(set(found_sections)),
            'bullet_points': bullet_count,
            'has_proper_sections': has_sections,
            'has_bullets': has_bullets
        }

    def calculate_readability(self, text):
        import textstat
        
        flesch_score = textstat.flesch_reading_ease(text)
        grade_level = textstat.flesch_kincaid_grade(text)
        
        return {
            'flesch_score': flesch_score,
            'grade_level': grade_level,
            'readability_rating': 'Good' if 60 <= flesch_score <= 90 else 'Needs Improvement'
        }

    def calculate_ats_score(self, resume_analysis):
        score = 0
        feedback = []
        
        # Contact Information (10 points)
        if resume_analysis['contact_info']['has_contact']:
            score += self.scoring_criteria['contact_info']
        else:
            feedback.append("Add complete contact information (email and phone)")
        
        # Keywords Match (25 points)
        total_keywords = sum(len(keywords) for keywords in resume_analysis['keywords']['found_keywords'].values())
        keyword_score = min(total_keywords * 2, self.scoring_criteria['keywords_match'])
        score += keyword_score
        if keyword_score < 15:
            feedback.append("Include more relevant technical and soft skills keywords")
        
        # Action Verbs (15 points)
        action_verb_count = len(resume_analysis['keywords']['found_keywords']['action_verbs'])
        action_score = min(action_verb_count * 3, self.scoring_criteria['action_verbs'])
        score += action_score
        if action_score < 10:
            feedback.append("Use more action verbs to describe your achievements")
        
        # Formatting (15 points)
        formatting = resume_analysis['formatting']
        format_score = 0
        if formatting['has_proper_sections']:
            format_score += 8
        if formatting['has_bullets']:
            format_score += 7
        score += format_score
        if format_score < 10:
            feedback.append("Improve formatting with clear sections and bullet points")
        
        # Length (10 points)
        word_count = len(resume_analysis['text'].split())
        if 400 <= word_count <= 800:
            score += self.scoring_criteria['length']
        elif word_count < 400:
            feedback.append("Resume is too short - add more details about your experience")
        else:
            feedback.append("Resume is too long - keep it concise (400-800 words)")
        
        # Sections (15 points)
        section_count = len(formatting['sections_found'])
        section_score = min(section_count * 5, self.scoring_criteria['sections'])
        score += section_score
        if section_score < 10:
            feedback.append("Include standard resume sections: Summary, Experience, Education, Skills")
        
        # Readability (10 points)
        readability = resume_analysis['readability']
        if readability['readability_rating'] == 'Good':
            score += self.scoring_criteria['readability']
        else:
            feedback.append("Improve readability - use simpler sentences and clearer language")
        
        return min(score, 100), feedback

    def generate_improvements(self, resume_analysis, ats_score):
        improvements = []
        
        # Keyword improvements
        missing_technical = [kw for kw in self.skill_keywords['technical'][:10] 
                           if kw not in resume_analysis['keywords']['found_keywords']['technical']]
        if missing_technical:
            improvements.append(f"Consider adding these technical skills if relevant: {', '.join(missing_technical[:5])}")
        
        # Action verb improvements
        if len(resume_analysis['keywords']['found_keywords']['action_verbs']) < 5:
            suggested_verbs = ['achieved', 'developed', 'implemented', 'managed', 'improved']
            improvements.append(f"Use more action verbs like: {', '.join(suggested_verbs)}")
        
        # Formatting improvements
        if not resume_analysis['formatting']['has_bullets']:
            improvements.append("Use bullet points to list your achievements and responsibilities")
        
        if len(resume_analysis['formatting']['sections_found']) < 4:
            improvements.append("Include standard sections: Professional Summary, Work Experience, Education, Skills")
        
        # Length improvements
        word_count = len(resume_analysis['text'].split())
        if word_count < 400:
            improvements.append("Expand your resume with more details about your accomplishments and quantify your results")
        elif word_count > 800:
            improvements.append("Reduce resume length by focusing on most relevant and recent experiences")
        
        return improvements

    def analyze_resume(self, text):
        if not text:
            return None
        
        analysis = {
            'text': text,
            'contact_info': self.extract_contact_info(text),
            'keywords': {},
            'formatting': self.check_formatting(text),
            'readability': self.calculate_readability(text)
        }
        
        found_keywords, keyword_density = self.analyze_keywords(text)
        analysis['keywords'] = {
            'found_keywords': found_keywords,
            'keyword_density': keyword_density
        }
        
        ats_score, feedback = self.calculate_ats_score(analysis)
        analysis['ats_score'] = ats_score
        analysis['feedback'] = feedback
        analysis['improvements'] = self.generate_improvements(analysis, ats_score)
        
        return analysis

def main():
    st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")
    
    st.title("🎯 ATS Resume Checker")
    st.markdown("Upload your resume to get an ATS compatibility score and improvement suggestions!")
    
    checker = ATSResumeChecker()
    
    if checker.nlp is None:
        st.stop()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx'],
            help="Upload a PDF or Word document"
        )
        
        if uploaded_file:
            st.success(f"File uploaded: {uploaded_file.name}")
    
    if uploaded_file:
        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = checker.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = checker.extract_text_from_docx(uploaded_file)
        else:
            st.error("Unsupported file type")
            st.stop()
        
        if text:
            # Analyze resume
            with st.spinner("Analyzing your resume..."):
                analysis = checker.analyze_resume(text)
            
            if analysis:
                # Display ATS Score
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    score_color = "red" if analysis['ats_score'] < 60 else "orange" if analysis['ats_score'] < 80 else "green"
                    st.metric("ATS Score", f"{analysis['ats_score']}/100")
                    st.markdown(f"<div style='color: {score_color}; font-weight: bold;'>{'Needs Improvement' if analysis['ats_score'] < 60 else 'Good' if analysis['ats_score'] < 80 else 'Excellent'}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Keywords Found", sum(len(keywords) for keywords in analysis['keywords']['found_keywords'].values()))
                
                with col3:
                    st.metric("Word Count", len(analysis['text'].split()))
                
                # Detailed Analysis
                st.header("📊 Detailed Analysis")
                
                # Keywords Analysis
                st.subheader("🔑 Keywords Analysis")
                keywords_data = []
                for category, keywords in analysis['keywords']['found_keywords'].items():
                    if keywords:
                        keywords_data.append({
                            'Category': category.replace('_', ' ').title(),
                            'Keywords Found': ', '.join(keywords),
                            'Count': len(keywords)
                        })
                
                if keywords_data:
                    df_keywords = pd.DataFrame(keywords_data)
                    st.dataframe(df_keywords, use_container_width=True)
                else:
                    st.warning("No relevant keywords found in your resume")
                
                # Formatting Analysis
                st.subheader("📝 Formatting Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Sections Found:**")
                    if analysis['formatting']['sections_found']:
                        for section in analysis['formatting']['sections_found']:
                            st.write(f"✅ {section.title()}")
                    else:
                        st.write("❌ No standard sections identified")
                
                with col2:
                    st.write("**Formatting Elements:**")
                    st.write(f"{'✅' if analysis['formatting']['has_bullets'] else '❌'} Bullet Points: {analysis['formatting']['bullet_points']}")
                    st.write(f"{'✅' if analysis['formatting']['has_proper_sections'] else '❌'} Proper Sections")
                    st.write(f"{'✅' if analysis['contact_info']['has_contact'] else '❌'} Complete Contact Info")
                
                # Readability
                st.subheader("📖 Readability")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Flesch Reading Score", f"{analysis['readability']['flesch_score']:.1f}")
                with col2:
                    st.metric("Grade Level", f"{analysis['readability']['grade_level']:.1f}")
                
                # Feedback and Improvements
                st.header("💡 Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚨 Issues to Fix")
                    if analysis['feedback']:
                        for i, feedback in enumerate(analysis['feedback'], 1):
                            st.write(f"{i}. {feedback}")
                    else:
                        st.success("Great! No major issues found.")
                
                with col2:
                    st.subheader("⬆️ Improvement Suggestions")
                    if analysis['improvements']:
                        for i, improvement in enumerate(analysis['improvements'], 1):
                            st.write(f"{i}. {improvement}")
                    else:
                        st.success("Your resume looks good!")
                
                # Word Cloud
                st.header("☁️ Word Cloud")
                if len(analysis['text']) > 100:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(analysis['text'])
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Download Report
                st.header("📥 Download Report")
                report_data = {
                    'ATS Score': analysis['ats_score'],
                    'Keywords Found': sum(len(keywords) for keywords in analysis['keywords']['found_keywords'].values()),
                    'Word Count': len(analysis['text'].split()),
                    'Sections Found': len(analysis['formatting']['sections_found']),
                    'Has Bullet Points': analysis['formatting']['has_bullets'],
                    'Readability Score': analysis['readability']['flesch_score'],
                    'Feedback': ' | '.join(analysis['feedback']),
                    'Improvements': ' | '.join(analysis['improvements'])
                }
                
                df_report = pd.DataFrame([report_data])
                csv = df_report.to_csv(index=False)
                
                st.download_button(
                    label="Download Analysis Report (CSV)",
                    data=csv,
                    file_name=f"ats_analysis_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
        else:
            st.error("Could not extract text from the uploaded file. Please try a different file.")
    
    else:
        # Instructions
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload your resume** (PDF or Word format) using the sidebar
        2. **Get your ATS score** and detailed analysis
        3. **Review feedback** and improvement suggestions
        4. **Download the report** for future reference
        
        ### What We Check:
        - **Contact Information**: Email and phone number presence
        - **Keywords**: Technical skills, soft skills, and action verbs
        - **Formatting**: Proper sections, bullet points, and structure
        - **Length**: Optimal word count (400-800 words)
        - **Readability**: How easy your resume is to read
        """)
        
        st.header("💡 ATS Tips")
        st.markdown("""
        - Use standard section headings (Experience, Education, Skills)
        - Include relevant keywords from job descriptions
        - Use bullet points for achievements
        - Keep formatting simple and clean
        - Quantify your accomplishments with numbers
        - Use action verbs to start bullet points
        """)

if __name__ == "__main__":
    main()