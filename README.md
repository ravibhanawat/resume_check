# ATS Resume Checker 🎯

A comprehensive Python application that analyzes resumes for ATS (Applicant Tracking System) compatibility, provides scoring, and offers improvement suggestions.

## Features

- **ATS Score Calculation**: Get a detailed score out of 100 based on ATS-friendly criteria
- **Keyword Analysis**: Identifies technical skills, soft skills, and action verbs
- **Formatting Check**: Analyzes resume structure, sections, and bullet points
- **Readability Assessment**: Evaluates how easy your resume is to read
- **Improvement Suggestions**: Provides actionable recommendations
- **Word Cloud Visualization**: Visual representation of your resume content
- **Report Export**: Download detailed analysis as CSV

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip3 install -r requirements.txt
```

3. Download the spaCy English model:

```bash
python3 -m spacy download en_core_web_sm
```

## Usage

1. Run the application:

```bash
python3 run_app.py
```

Or directly with streamlit:

```bash
python3 -m streamlit run ats_resume_checker.py
```

2. Open your browser and navigate to the displayed URL (usually `http://localhost:8501`)

3. Upload your resume (PDF or Word format) using the sidebar

4. View your ATS score and detailed analysis

5. Follow the improvement suggestions to optimize your resume

## What Gets Analyzed

### Scoring Criteria (Total: 100 points)

- **Contact Information (10 points)**: Email and phone number presence
- **Keywords Match (25 points)**: Relevant technical and soft skills
- **Action Verbs (15 points)**: Use of strong action verbs
- **Formatting (15 points)**: Proper sections and bullet points
- **Length (10 points)**: Optimal word count (400-800 words)
- **Sections (15 points)**: Standard resume sections
- **Readability (10 points)**: Text clarity and simplicity

### Key Features Analyzed

1. **Contact Information**
   - Email address detection
   - Phone number detection

2. **Keywords**
   - Technical skills (Python, Java, AWS, etc.)
   - Soft skills (Leadership, Communication, etc.)
   - Action verbs (Developed, Managed, etc.)

3. **Formatting**
   - Standard sections (Experience, Education, Skills)
   - Bullet point usage
   - Overall structure

4. **Readability**
   - Flesch Reading Ease score
   - Grade level assessment

## Sample Resume Sections Detected

- Experience / Work Experience
- Education
- Skills / Technical Skills
- Summary / Professional Summary
- Objective / Career Objective
- Projects

## Tips for Better ATS Scores

1. **Use Standard Headings**: Experience, Education, Skills, Summary
2. **Include Keywords**: Match job description keywords
3. **Use Bullet Points**: List achievements and responsibilities
4. **Keep It Simple**: Avoid complex formatting, tables, or graphics
5. **Quantify Results**: Use numbers to show impact
6. **Action Verbs**: Start bullet points with strong action verbs
7. **Optimal Length**: Keep resume between 400-800 words
8. **Contact Info**: Always include email and phone number

## Supported File Formats

- PDF (.pdf)
- Microsoft Word (.docx)

## Requirements

- Python 3.8+
- All dependencies listed in `requirements.txt`
- spaCy English model (`en_core_web_sm`)

## Technical Details

The application uses:
- **PyPDF2** for PDF text extraction
- **python-docx** for Word document processing
- **NLTK** for natural language processing
- **spaCy** for advanced text analysis
- **Streamlit** for the web interface
- **scikit-learn** for text similarity analysis
- **matplotlib/seaborn** for visualizations

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting new features
- Improving the scoring algorithm
- Adding new keyword categories
- Enhancing the UI/UX