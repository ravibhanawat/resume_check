from setuptools import setup, find_packages

setup(
    name="ats-resume-checker",
    version="1.0.0",
    description="ATS Resume Checker - Analyze resume compatibility with Applicant Tracking Systems",
    author="Resume Checker",
    packages=find_packages(),
    install_requires=[
        "PyPDF2>=3.0.1",
        "python-docx>=0.8.11",
        "nltk>=3.8.1",
        "spacy>=3.7.0",
        "textstat>=0.7.3",
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "wordcloud>=1.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)