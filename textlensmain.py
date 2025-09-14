"""
TextLens - Comprehensive Text Analysis Bot
Main analyzer module for linguistic analysis
"""

import re
import string
from collections import Counter
from typing import Dict, List, Tuple, Any
import nltk
from textblob import TextBlob
from textstat import flesch_reading_ease, flesch_kincaid_grade, gunning_fog, automated_readability_index
from langdetect import detect
import spacy

# Download required NLTK data (run once)
def setup_nltk():
    """Download required NLTK data packages"""
    required_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'maxent_ne_chunker', 'words']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

class TextAnalyzer:
    """Main text analysis class with comprehensive linguistic metrics"""
    
    def __init__(self, use_spacy=False):
        """
        Initialize the analyzer
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP (requires model download)
        """
        setup_nltk()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        self.use_spacy = use_spacy
        
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    def analyze(self, text: str, include_stopwords: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive text analysis
        
        Args:
            text: Input text to analyze
            include_stopwords: Whether to include stop words in frequency analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        results = {}
        
        # Basic preprocessing
        clean_text = self._preprocess_text(text)
        words = self._tokenize_words(clean_text)
        sentences = nltk.sent_tokenize(text)
        
        # Core metrics
        results['basic_metrics'] = self._calculate_basic_metrics(text, words, sentences)
        
        # Word frequency analysis
        results['word_frequency'] = self._analyze_word_frequency(words, include_stopwords)
        
        # Sentiment analysis
        results['sentiment'] = self._analyze_sentiment(text)
        
        # Readability scores
        results['readability'] = self._calculate_readability(text)
        
        # Lexical diversity
        results['lexical_diversity'] = self._calculate_lexical_diversity(words)
        
        # Part of speech distribution
        results['pos_distribution'] = self._analyze_pos_distribution(words)
        
        # Language detection
        results['language'] = self._detect_language(text)
        
        # Advanced NLP features (if spaCy is available)
        if self.use_spacy:
            results['named_entities'] = self._extract_named_entities(text)
        
        # Find longest word
        results['longest_word'] = self._find_longest_word(words)
        
        # Calculate syllables and complex words
        results['syllable_metrics'] = self._calculate_syllable_metrics(words)
        
        return results
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Remove punctuation and convert to lowercase
        translator = str.maketrans('', '', string.punctuation)
        text_no_punct = text.translate(translator).lower()
        return nltk.word_tokenize(text_no_punct)
    
    def _calculate_basic_metrics(self, text: str, words: List[str], sentences: List[str]) -> Dict:
        """Calculate basic text metrics"""
        return {
            'total_characters': len(text),
            'total_characters_no_spaces': len(text.replace(' ', '')),
            'total_words': len(words),
            'unique_words': len(set(words)),
            'total_sentences': len(sentences),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def _analyze_word_frequency(self, words: List[str], include_stopwords: bool) -> Dict:
        """Analyze word frequency"""
        if not include_stopwords:
            words = [w for w in words if w not in self.stop_words]
        
        word_freq = Counter(words)
        top_50 = dict(word_freq.most_common(50))
        
        return {
            'top_50_words': top_50,
            'total_unique_words': len(word_freq),
            'most_common_word': word_freq.most_common(1)[0] if word_freq else None
        }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Perform sentiment analysis"""
        blob = TextBlob(text)
        
        # Polarity: -1 (negative) to 1 (positive)
        # Subjectivity: 0 (objective) to 1 (subjective)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Categorize sentiment
        if polarity > 0.1:
            sentiment_label = "positive"
        elif polarity < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            'polarity_score': round(polarity, 3),
            'subjectivity_score': round(subjectivity, 3),
            'sentiment_label': sentiment_label,
            'positivity_percentage': round((polarity + 1) * 50, 2)  # Convert to 0-100 scale
        }
    
    def _calculate_readability(self, text: str) -> Dict:
        """Calculate various readability scores"""
        try:
            return {
                'flesch_reading_ease': round(flesch_reading_ease(text), 2),
                'flesch_kincaid_grade': round(flesch_kincaid_grade(text), 2),
                'gunning_fog': round(gunning_fog(text), 2),
                'automated_readability_index': round(automated_readability_index(text), 2),
                'average_grade_level': round(
                    (flesch_kincaid_grade(text) + gunning_fog(text) + automated_readability_index(text)) / 3, 2
                )
            }
        except:
            return {'error': 'Text too short for readability analysis'}
    
    def _calculate_lexical_diversity(self, words: List[str]) -> Dict:
        """Calculate lexical diversity metrics"""
        if not words:
            return {'type_token_ratio': 0, 'lexical_diversity': 0}
        
        unique_words = set(words)
        ttr = len(unique_words) / len(words)
        
        # Root Type-Token Ratio (more stable for longer texts)
        rttr = len(unique_words) / (len(words) ** 0.5) if words else 0
        
        return {
            'type_token_ratio': round(ttr, 3),
            'root_ttr': round(rttr, 3),
            'lexical_diversity_percentage': round(ttr * 100, 2)
        }
    
    def _analyze_pos_distribution(self, words: List[str]) -> Dict:
        """Analyze part-of-speech distribution"""
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Group into major categories
        categories = {
            'nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
            'verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adjectives': ['JJ', 'JJR', 'JJS'],
            'adverbs': ['RB', 'RBR', 'RBS'],
            'pronouns': ['PRP', 'PRP$', 'WP', 'WP$'],
            'determiners': ['DT', 'PDT', 'WDT'],
            'prepositions': ['IN', 'TO'],
            'conjunctions': ['CC'],
            'modals': ['MD']
        }
        
        distribution = {}
        for category, tags in categories.items():
            count = sum(pos_counts.get(tag, 0) for tag in tags)
            distribution[category] = {
                'count': count,
                'percentage': round((count / len(words) * 100), 2) if words else 0
            }
        
        return distribution
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text"""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def _extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities using spaCy"""
        if not self.use_spacy:
            return []
        
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities[:20]  # Return top 20 entities
    
    def _find_longest_word(self, words: List[str]) -> Dict:
        """Find the longest word(s) in the text"""
        if not words:
            return {'word': '', 'length': 0}
        
        max_length = max(len(word) for word in words)
        longest_words = [word for word in words if len(word) == max_length]
        
        return {
            'word': longest_words[0],
            'length': max_length,
            'all_longest_words': list(set(longest_words))
        }
    
    def _calculate_syllable_metrics(self, words: List[str]) -> Dict:
        """Calculate syllable-based metrics"""
        def count_syllables(word):
            """Simple syllable counter"""
            word = word.lower()
            count = 0
            vowels = 'aeiouy'
            if word[0] in vowels:
                count += 1
            for index in range(1, len(word)):
                if word[index] in vowels and word[index - 1] not in vowels:
                    count += 1
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count += 1
            return count
        
        syllable_counts = [count_syllables(word) for word in words if word]
        complex_words = [word for word in words if count_syllables(word) >= 3]
        
        return {
            'avg_syllables_per_word': round(sum(syllable_counts) / len(syllable_counts), 2) if syllable_counts else 0,
            'complex_words_count': len(complex_words),
            'complex_words_percentage': round((len(complex_words) / len(words) * 100), 2) if words else 0
        }
    
    def generate_report(self, analysis_results: Dict) -> str:
        """Generate a formatted text report from analysis results"""
        report = []
        report.append("=" * 60)
        report.append("TEXT ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Basic Metrics
        if 'basic_metrics' in analysis_results:
            report.append("\n📊 BASIC METRICS:")
            metrics = analysis_results['basic_metrics']
            report.append(f"  • Total Words: {metrics['total_words']:,}")
            report.append(f"  • Unique Words: {metrics['unique_words']:,}")
            report.append(f"  • Total Sentences: {metrics['total_sentences']:,}")
            report.append(f"  • Avg Word Length: {metrics['avg_word_length']:.1f} characters")
            report.append(f"  • Avg Sentence Length: {metrics['avg_sentence_length']:.1f} words")
        
        # Sentiment Analysis
        if 'sentiment' in analysis_results:
            report.append("\n😊 SENTIMENT ANALYSIS:")
            sentiment = analysis_results['sentiment']
            report.append(f"  • Overall: {sentiment['sentiment_label'].upper()}")
            report.append(f"  • Positivity: {sentiment['positivity_percentage']:.1f}%")
            report.append(f"  • Polarity Score: {sentiment['polarity_score']}")
            report.append(f"  • Subjectivity: {sentiment['subjectivity_score']}")
        
        # Readability
        if 'readability' in analysis_results:
            report.append("\n📖 READABILITY:")
            readability = analysis_results['readability']
            if 'error' not in readability:
                report.append(f"  • Average Grade Level: {readability['average_grade_level']}")
                report.append(f"  • Flesch Reading Ease: {readability['flesch_reading_ease']}")
                report.append(f"  • Flesch-Kincaid Grade: {readability['flesch_kincaid_grade']}")
        
        # Longest Word
        if 'longest_word' in analysis_results:
            report.append("\n📏 LONGEST WORD:")
            longest = analysis_results['longest_word']
            report.append(f"  • Word: '{longest['word']}'")
            report.append(f"  • Length: {longest['length']} characters")
        
        # Top Words
        if 'word_frequency' in analysis_results:
            report.append("\n🔤 TOP 10 MOST COMMON WORDS:")
            top_words = list(analysis_results['word_frequency']['top_50_words'].items())[:10]
            for i, (word, count) in enumerate(top_words, 1):
                report.append(f"  {i:2}. '{word}' - {count} times")
        
        # Lexical Diversity
        if 'lexical_diversity' in analysis_results:
            report.append("\n🎨 LEXICAL DIVERSITY:")
            diversity = analysis_results['lexical_diversity']
            report.append(f"  • Vocabulary Richness: {diversity['lexical_diversity_percentage']:.1f}%")
            report.append(f"  • Type-Token Ratio: {diversity['type_token_ratio']}")
        
        # Language
        if 'language' in analysis_results:
            report.append(f"\n🌐 DETECTED LANGUAGE: {analysis_results['language'].upper()}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
