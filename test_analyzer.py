"""
Unit tests for TextLens analyzer
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analyzer import TextAnalyzer

class TestTextAnalyzer:
    """Test suite for TextAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for tests"""
        return TextAnalyzer(use_spacy=False)
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing"""
        return """
        The quick brown fox jumps over the lazy dog. 
        This is a simple test sentence with various words. 
        Testing is important for quality assurance.
        """
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer is not None
        assert analyzer.stop_words is not None
        assert len(analyzer.stop_words) > 0
    
    def test_empty_text_handling(self, analyzer):
        """Test handling of empty text"""
        result = analyzer.analyze("")
        assert "error" in result
        
        result = analyzer.analyze("   ")
        assert "error" in result
    
    def test_basic_metrics(self, analyzer, sample_text):
        """Test basic metric calculations"""
        results = analyzer.analyze(sample_text)
        
        assert 'basic_metrics' in results
        metrics = results['basic_metrics']
        
        assert metrics['total_words'] > 0
        assert metrics['unique_words'] > 0
        assert metrics['total_sentences'] == 3
        assert metrics['avg_word_length'] > 0
        assert metrics['avg_sentence_length'] > 0
    
    def test_word_frequency(self, analyzer, sample_text):
        """Test word frequency analysis"""
        results = analyzer.analyze(sample_text)
        
        assert 'word_frequency' in results
        freq = results['word_frequency']
        
        assert 'top_50_words' in freq
        assert isinstance(freq['top_50_words'], dict)
        assert len(freq['top_50_words']) > 0
        assert freq['total_unique_words'] > 0
    
    def test_sentiment_analysis(self, analyzer):
        """Test sentiment analysis"""
        positive_text = "I love this amazing product! It's wonderful and fantastic!"
        negative_text = "This is terrible, awful, and horrible. I hate it."
        neutral_text = "The sky is blue. Water is wet. Facts are facts."
        
        # Test positive sentiment
        pos_results = analyzer.analyze(positive_text)
        assert pos_results['sentiment']['polarity_score'] > 0
        assert pos_results['sentiment']['sentiment_label'] == 'positive'
        
        # Test negative sentiment
        neg_results = analyzer.analyze(negative_text)
        assert neg_results['sentiment']['polarity_score'] < 0
        assert neg_results['sentiment']['sentiment_label'] == 'negative'
        
        # Test neutral sentiment
        neut_results = analyzer.analyze(neutral_text)
        assert abs(neut_results['sentiment']['polarity_score']) < 0.2
    
    def test_readability_scores(self, analyzer):
        """Test readability calculations"""
        # Simple text
        simple_text = "The cat sat on the mat. The dog ran fast. Birds fly high."
        simple_results = analyzer.analyze(simple_text)
        
        if 'error' not in simple_results['readability']:
            assert simple_results['readability']['flesch_reading_ease'] > 60
            assert simple_results['readability']['flesch_kincaid_grade'] < 8
        
        # Complex text
        complex_text = """
        The implementation of sophisticated algorithmic paradigms necessitates 
        comprehensive understanding of computational complexity theory and 
        mathematical optimization techniques, particularly in the context of 
        distributed systems architecture and concurrent processing methodologies.
        """
        complex_results = analyzer.analyze(complex_text)
        
        if 'error' not in complex_results['readability']:
            assert complex_results['readability']['flesch_reading_ease'] < 30
            assert complex_results['readability']['flesch_kincaid_grade'] > 15
    
    def test_longest_word(self, analyzer):
        """Test longest word detection"""
        text = "Short words and an extraordinarily long word here."
        results = analyzer.analyze(text)
        
        assert 'longest_word' in results
        assert results['longest_word']['word'] == 'extraordinarily'
        assert results['longest_word']['length'] == 15
    
    def test_lexical_diversity(self, analyzer):
        """Test lexical diversity metrics"""
        # High diversity text (all unique words)
        diverse_text = "cat dog bird fish tree flower sun moon star"
        diverse_results = analyzer.analyze(diverse_text)
        assert diverse_results['lexical_diversity']['type_token_ratio'] == 1.0
        
        # Low diversity text (repeated words)
        repetitive_text = "test test test test test word word word word"
        rep_results = analyzer.analyze(repetitive_text)
        assert rep_results['lexical_diversity']['type_token_ratio'] < 0.3
    
    def test_pos_distribution(self, analyzer, sample_text):
        """Test part-of-speech distribution"""
        results = analyzer.analyze(sample_text)
        
        assert 'pos_distribution' in results
        pos = results['pos_distribution']
        
        # Check that main categories exist
        assert 'nouns' in pos
        assert 'verbs' in pos
        assert 'adjectives' in pos
        
        # Check structure
        assert 'count' in pos['nouns']
        assert 'percentage' in pos['nouns']
    
    def test_language_detection(self, analyzer):
        """Test language detection"""
        english_text = "This is an English sentence."
        results = analyzer.analyze(english_text)
        assert results['language'] == 'en'
    
    def test_syllable_metrics(self, analyzer, sample_text):
        """Test syllable calculations"""
        results = analyzer.analyze(sample_text)
        
        assert 'syllable_metrics' in results
        syllables = results['syllable_metrics']
        
        assert syllables['avg_syllables_per_word'] > 0
        assert syllables['complex_words_count'] >= 0
        assert syllables['complex_words_percentage'] >= 0
    
    def test_stopwords_filtering(self, analyzer):
        """Test stopword filtering in word frequency"""
        text = "The the the cat and the dog"
        
        # With stopwords
        with_stop = analyzer.analyze(text, include_stopwords=True)
        top_word = list(with_stop['word_frequency']['top_50_words'].keys())[0]
        assert top_word == 'the'
        
        # Without stopwords
        without_stop = analyzer.analyze(text, include_stopwords=False)
        top_words = list(without_stop['word_frequency']['top_50_words'].keys())
        assert 'the' not in top_words
        assert 'cat' in top_words or 'dog' in top_words
    
    def test_report_generation(self, analyzer, sample_text):
        """Test report generation"""
        results = analyzer.analyze(sample_text)
        report = analyzer.generate_report(results)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "TEXT ANALYSIS REPORT" in report
        assert "BASIC METRICS" in report
        assert "SENTIMENT ANALYSIS" in report

    def test_preprocessing(self, analyzer):
        """Test text preprocessing"""
        text_with_urls = "Check out https://example.com and email me at test@email.com"
        results = analyzer.analyze(text_with_urls)
        
        # URLs and emails should be removed
        words = results['basic_metrics']['total_words']
        assert words < 10  # Should have fewer words after preprocessing
    
    def test_edge_cases(self, analyzer):
        """Test various edge cases"""
        # Single word
        single = analyzer.analyze("Hello")
        assert single['basic_metrics']['total_words'] == 1
        
        # Numbers only
        numbers = analyzer.analyze("123 456 789")
        assert numbers['basic_metrics']['total_words'] == 3
        
        # Special characters
        special = analyzer.analyze("Hello! @#$% World?")
        assert special['basic_metrics']['total_words'] == 2
        
        # Very long word
        long_word = "pneumonoultramicroscopicsilicovolcanoconiosis"
        long_text = f"The disease {long_word} is complicated"
        long_results = analyzer.analyze(long_text)
        assert long_results['longest_word']['word'] == long_word

@pytest.mark.parametrize("text,expected_grade", [
    ("The cat sat.", 0),  # Very simple
    ("The implementation requires analysis.", 10),  # Moderate
])
def test_readability_ranges(text, expected_grade):
    """Test readability score ranges"""
    analyzer = TextAnalyzer()
    results = analyzer.analyze(text * 10)  # Repeat for minimum length
    
    if 'error' not in results['readability']:
        grade = results['readability']['flesch_kincaid_grade']
        # Allow for some variance
        assert abs(grade - expected_grade) < 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
