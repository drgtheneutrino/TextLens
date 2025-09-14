# TextLens 🔍

A comprehensive text analysis bot that performs deep linguistic analysis on any text input. TextLens provides detailed insights into text characteristics, readability, sentiment, and linguistic patterns.

## Features ✨

### Core Analysis Capabilities
- **📊 Word Frequency Analysis**: Top 50 most common words with optional stopword filtering
- **📏 Lexical Metrics**: Longest word, unique word count, vocabulary richness
- **😊 Sentiment Analysis**: Positivity/negativity scoring with detailed polarity metrics
- **📖 Readability Scores**: Multiple grade-level formulas (Flesch-Kincaid, Gunning Fog, ARI)
- **🎨 Lexical Diversity**: Type-Token Ratio and vocabulary richness metrics
- **🔤 Part-of-Speech Distribution**: Detailed grammatical composition analysis
- **🌐 Language Detection**: Automatic language identification
- **📝 Named Entity Recognition**: Extract people, places, organizations (optional with spaCy)

### Additional Metrics
- Average word and sentence length
- Syllable counting and complex word identification
- Character and word count statistics
- Subjectivity analysis
- Multiple readability indices

## Installation 🚀

### Basic Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/textlens.git
cd textlens

# Install required dependencies
pip install -r requirements.txt

# Download NLTK data (run Python and execute)
python -c "import nltk; nltk.download(['punkt', 'averaged_perceptron_tagger', 'stopwords', 'maxent_ne_chunker', 'words'])"
```

### Optional: Advanced NLP with spaCy
```bash
# Install spaCy and download English model
pip install spacy
python -m spacy download en_core_web_sm
```

## Quick Start 🏃‍♂️

### Basic Usage
```python
from analyzer import TextAnalyzer

# Initialize the analyzer
analyzer = TextAnalyzer()

# Analyze text
text = "Your text here..."
results = analyzer.analyze(text)

# Generate a formatted report
report = analyzer.generate_report(results)
print(report)
```

### Run the Example Script
```bash
cd examples
python sample_analysis.py
```

## API Reference 📚

### TextAnalyzer Class

#### `__init__(use_spacy=False)`
Initialize the analyzer with optional spaCy support for advanced NLP features.

#### `analyze(text, include_stopwords=False)`
Perform comprehensive text analysis.

**Parameters:**
- `text` (str): Input text to analyze
- `include_stopwords` (bool): Whether to include stop words in frequency analysis

**Returns:**
Dictionary containing:
- `basic_metrics`: Word counts, sentence counts, averages
- `word_frequency`: Top 50 words and frequency data
- `sentiment`: Polarity, subjectivity, and sentiment label
- `readability`: Multiple readability scores and grade levels
- `lexical_diversity`: TTR and vocabulary richness metrics
- `pos_distribution`: Part-of-speech breakdown
- `longest_word`: Longest word(s) in the text
- `syllable_metrics`: Syllable counts and complex word analysis
- `language`: Detected language code

#### `generate_report(analysis_results)`
Generate a formatted text report from analysis results.

## Output Example 📋

```
===========================================================
TEXT ANALYSIS REPORT
===========================================================

📊 BASIC METRICS:
  • Total Words: 156
  • Unique Words: 98
  • Total Sentences: 8
  • Avg Word Length: 5.2 characters
  • Avg Sentence Length: 19.5 words

😊 SENTIMENT ANALYSIS:
  • Overall: POSITIVE
  • Positivity: 65.3%
  • Polarity Score: 0.306
  • Subjectivity: 0.412

📖 READABILITY:
  • Average Grade Level: 12.4
  • Flesch Reading Ease: 42.3
  • Flesch-Kincaid Grade: 11.8

📏 LONGEST WORD:
  • Word: 'implementation'
  • Length: 14 characters

🔤 TOP 10 MOST COMMON WORDS:
   1. 'technology' - 8 times
   2. 'analysis' - 6 times
   3. 'data' - 5 times
   ...

🎨 LEXICAL DIVERSITY:
  • Vocabulary Richness: 62.8%
  • Type-Token Ratio: 0.628

🌐 DETECTED LANGUAGE: EN
===========================================================
```

## Use Cases 💡

- **Content Analysis**: Analyze blog posts, articles, or documents for readability and engagement
- **Academic Writing**: Check grade level and complexity of academic papers
- **Social Media Monitoring**: Sentiment analysis for brand monitoring
- **SEO Optimization**: Analyze keyword frequency and content quality
- **Language Learning**: Assess text difficulty for language learners
- **Editorial Tools**: Evaluate writing style and vocabulary diversity
- **Research**: Linguistic analysis for research projects

## Testing 🧪

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Future Enhancements 🔮

- [ ] Web interface with Flask/FastAPI
- [ ] PDF and DOCX file support
- [ ] Multi-language support
- [ ] Emotion detection (beyond positive/negative)
- [ ] Topic modeling and keyword extraction
- [ ] Comparative analysis between multiple texts
- [ ] Export reports to PDF/HTML
- [ ] Real-time analysis API endpoint
- [ ] Machine learning-based style detection
- [ ] Plagiarism detection features

## License 📄

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- NLTK for natural language processing tools
- TextBlob for sentiment analysis
- TextStat for readability calculations
- spaCy for advanced NLP features

## Contact 📧

Your Name - [@yourusername](https://twitter.com/yourusername)

Project Link: [https://github.com/yourusername/textlens](https://github.com/yourusername/textlens)

---

Made with ❤️ by [Your Name]
