# from underthesea import word_tokenize

# def vi_tokenizer(text: str):
#     # Trả về list token đã tách từ, có thể nối cụm bằng dấu gạch dưới
#     return word_tokenize(text, format="text").split()

# -*- coding: utf-8 -*-
# """Test Vietnamese tokenizer"""

# from underthesea import word_tokenize

# def vi_tokenizer(text: str):
#     """Tokenize Vietnamese text"""
#     return word_tokenize(text, format="text").split()

# # Test text
# results = vi_tokenizer("""""")

# -*- coding: utf-8 -*-
"""Test Vietnamese tokenizer"""

from underthesea import word_tokenize

def vi_tokenizer(text: str):
    """Tokenize Vietnamese text"""
    return word_tokenize(text, format="text").split()

# Test text

 
results = vi_tokenizer(text)
print(f"Tokens: {len(results)}")
print(f"First 20 tokens: {results[:20]}")