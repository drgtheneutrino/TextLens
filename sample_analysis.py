#!/usr/bin/env python3
"""
Example usage of the TextLens analyzer
"""

import json
from pathlib import Path
import sys

# Add parent directory to path to import the analyzer
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analyzer import TextAnalyzer

def main():
    # Sample texts for demonstration
    sample_texts = {
        "news_article": """
        Artificial intelligence continues to revolutionize various industries across the globe. 
        From healthcare to finance, AI-powered solutions are transforming how businesses operate 
        and deliver services. Machine learning algorithms can now diagnose diseases with remarkable 
        accuracy, predict market trends, and even create art. However, experts warn about the 
        potential risks and ethical challenges that come with rapid AI advancement. The need for 
        responsible AI development and proper regulation has never been more critical as we navigate 
        this technological revolution.
        """,
        
        "literary": """
        In the quiet moments between dusk and dawn, when shadows dance with whispered secrets, 
        the old lighthouse keeper maintained his solitary vigil. The weathered stones beneath 
        his feet had witnessed countless storms, yet they stood firm against the relentless 
        assault of time and tide. Each beam of light that pierced the darkness carried with it 
        a promise - a beacon of hope for those lost in the tempestuous seas of uncertainty.
        """,
        
        "technical": """
        The implementation of quantum computing algorithms requires careful consideration of 
        quantum entanglement and superposition principles. Qubits, unlike classical bits, can 
        exist in multiple states simultaneously, enabling exponential computational advantages 
        for specific problem domains. Current challenges include maintaining quantum coherence, 
        error correction, and scaling to practical system sizes. Researchers are exploring 
        various approaches including superconducting circuits, trapped ions, and topological 
        qubits to overcome these limitations.
        """
    }
    
    # Initialize analyzer
    print("Initializing TextLens Analyzer...")
    analyzer = TextAnalyzer(use_spacy=False)  # Set to True if spaCy is installed
    
    # Analyze each sample text
    for text_type, text in sample_texts.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {text_type.upper()} text...")
        print(f"{'='*60}")
        
        # Perform analysis
        results = analyzer.analyze(text)
        
        # Generate and print report
        report = analyzer.generate_report(results)
        print(report)
        
        # Save detailed results to JSON
        output_file = f"analysis_{text_type}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter your own text for analysis (or 'quit' to exit):")
    
    while True:
        print("\n> ", end="")
        user_text = input()
        
        if user_text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if len(user_text.strip()) < 10:
            print("Please enter a longer text for meaningful analysis.")
            continue
        
        # Analyze user's text
        results = analyzer.analyze(user_text)
        report = analyzer.generate_report(results)
        print("\n" + report)
        
        # Ask if user wants to save results
        save = input("\nSave detailed results to JSON? (y/n): ")
        if save.lower() == 'y':
            filename = input("Enter filename (without .json): ")
            with open(f"{filename}.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {filename}.json")

if __name__ == "__main__":
    main()
