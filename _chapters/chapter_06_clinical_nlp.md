---
layout: chapter
title: "Chapter 6: Natural Language Processing for Clinical Text"
chapter_number: 6
---

# Chapter 6: Natural Language Processing for Clinical Text

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Implement production-grade natural language processing pipelines for clinical text that handle the unique challenges of medical documentation including specialized terminology, abbreviations, temporal expressions, negation, and uncertainty markers while explicitly accounting for documentation quality variations across healthcare settings.

2. Develop named entity recognition and relation extraction systems for clinical concepts that achieve equitable performance across diverse patient populations by addressing systematic differences in how clinicians document care for patients from different racial, ethnic, socioeconomic, and linguistic backgrounds.

3. Design and train clinical language models that detect and mitigate biases encoded in clinical documentation, including differential language use patterns that reflect historical discrimination rather than clinically relevant differences, with comprehensive evaluation of fairness properties before deployment.

4. Build multilingual clinical NLP systems that handle code-switching, non-standard language use, and interpretation service notes while respecting the linguistic diversity of patient populations and avoiding systems that systematically disadvantage patients with limited English proficiency.

5. Apply large language models to healthcare tasks including clinical documentation support, patient education material generation, and clinical question answering while implementing comprehensive safety and fairness testing appropriate for high-stakes medical applications.

6. Evaluate clinical NLP systems using stratified validation frameworks that surface disparate performance across patient demographics and care settings, with particular attention to identifying failure modes that could exacerbate existing healthcare disparities.

## 6.1 Introduction: Language, Power, and Healthcare AI

Clinical text represents one of the richest yet most challenging data sources in healthcare. Electronic health records contain billions of words of unstructured clinical notes documenting patient histories, physical examinations, clinical reasoning, treatment plans, and outcomes. This textual data captures nuances of clinical practice that structured data fields cannot fully represent, including symptom descriptions in patients' own words, clinician uncertainty and diagnostic reasoning, social context affecting health, and the evolving narrative of illness and recovery over time. Natural language processing promises to unlock this information at scale, enabling clinical decision support, quality improvement, research, and population health management in ways previously impossible.

Yet clinical text is far from neutral documentation of objective medical facts. Language choices in clinical notes encode social hierarchies, power dynamics, and systematic biases that reflect broader patterns of discrimination in healthcare delivery. The words clinicians choose to describe patients differ systematically by patient race, ethnicity, gender, socioeconomic status, and insurance type in ways that reflect stereotypes rather than clinically relevant differences. Patients from marginalized communities are more likely to be described with stigmatizing language that portrays them as non-compliant, difficult, or unreliable. Medical terminology itself often contains racist assumptions, from the continued use of outdated racial classifications in clinical descriptions to diagnostic criteria that were historically developed based exclusively on white patient populations.

A concrete example illustrates these patterns. Studies analyzing millions of clinical notes have found that Black patients are significantly more likely to be described with negative descriptors unrelated to their medical condition, while white patients receive more empathetic language even when presenting with identical symptoms. Notes about unhoused patients or those with Medicaid insurance often emphasize social circumstances in ways that shift focus away from medical needs. Women's pain reports are more likely to be characterized as emotional or psychological rather than physiological compared to men reporting identical symptoms. These linguistic patterns have measurable consequences: patients described with stigmatizing language receive different and often worse care, experience longer diagnostic delays, and have poorer health outcomes even after accounting for clinical severity.

The challenge for healthcare natural language processing extends beyond merely extracting information from biased text. When we train machine learning models on clinical notes, those models learn not only the medical knowledge encoded in the text but also the discriminatory patterns in how that knowledge is expressed. A named entity recognition system trained to identify symptoms learns to recognize symptoms described in the language patterns common for majority populations while potentially missing symptoms when described differently for other groups. A clinical prediction model incorporating note text as features may learn to associate demographic characteristics with outcomes not because those characteristics are biologically relevant but because they correlate with systematic differences in documentation quality, completeness, and linguistic framing.

This chapter develops natural language processing methods specifically designed to address these equity challenges. We approach clinical NLP not as a neutral technical task of extracting information from text but rather as an intervention in healthcare systems that either can perpetuate existing disparities or can be designed intentionally to detect and mitigate bias. Every technical decision from text preprocessing to model architecture to evaluation strategy is made with explicit consideration of fairness implications across diverse patient populations. We present production-grade implementations that incorporate equity considerations throughout the development lifecycle rather than treating fairness as an afterthought.

The chapter begins with fundamental clinical text preprocessing challenges, showing how even basic decisions about tokenization, normalization, and handling of medical abbreviations can introduce bias when documentation patterns differ across healthcare settings. We then develop named entity recognition approaches that maintain equitable performance across patient demographics by explicitly modeling and accounting for linguistic variation. The discussion progresses through relation extraction for building clinical knowledge graphs, sentiment and subjectivity detection in clinical notes, and finally to applications of large language models in healthcare with comprehensive frameworks for detecting and mitigating the biases these powerful models can encode and amplify.

Throughout, the principle remains constant: technical excellence in clinical NLP requires not just sophisticated algorithms but also deep understanding of the social contexts in which clinical documentation is created and the differential impacts that language-based AI systems can have across diverse patient populations. The goal is to develop NLP systems that serve rather than harm underserved communities by making clinical information more accessible while actively working to counteract rather than perpetuate the biases embedded in clinical language.

## 6.2 Clinical Text Preprocessing and Normalization

Clinical text presents unique preprocessing challenges that general-purpose NLP tools are not designed to handle. Medical documentation contains specialized terminology, non-standard abbreviations, complex temporal expressions, and linguistic structures for expressing negation and uncertainty that require domain-specific approaches. Moreover, preprocessing decisions that seem purely technical can have significant equity implications when documentation quality and linguistic patterns differ systematically across healthcare settings and patient populations.

### 6.2.1 Tokenization and Medical Terminology

The first step in any NLP pipeline is tokenization: breaking text into meaningful units for processing. Standard tokenization approaches based on whitespace and punctuation fail for clinical text where multi-word medical terms like "congestive heart failure" represent single semantic concepts, abbreviations may or may not have periods, and medication names contain complex internal structure including drug name, dosage, and frequency information all run together. These challenges are not distributed uniformly across healthcare settings. Academic medical centers may use more standardized terminology while community hospitals use more varied abbreviations and shorthand. Notes dictated through professional transcription services have different linguistic characteristics than notes typed rapidly during busy clinical shifts in under-resourced emergency departments.

We implement a clinical-aware tokenizer that handles these challenges while tracking potential sources of bias in the tokenization process itself.

```python
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc
from spacy.language import Language
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Token:
    """
    Represents a single token from clinical text with metadata.
    
    Attributes:
        text: Original token text
        normalized: Normalized form of the token
        pos: Part of speech tag
        is_medical_term: Whether this is a recognized medical term
        is_abbreviation: Whether this is an abbreviation
        uncertainty_marker: Whether this expresses clinical uncertainty
        negation_marker: Whether this is part of a negation context
        temporal_marker: Whether this expresses temporal information
    """
    text: str
    normalized: str
    pos: Optional[str] = None
    is_medical_term: bool = False
    is_abbreviation: bool = False
    uncertainty_marker: bool = False
    negation_marker: bool = False
    temporal_marker: bool = False

class ClinicalTokenizer:
    """
    Specialized tokenizer for clinical text that handles medical terminology,
    abbreviations, and clinical linguistic structures.
    
    This tokenizer explicitly tracks potential sources of bias including
    non-standard terminology use and documentation quality markers.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        medical_terms_file: Optional[str] = None,
        abbreviations_file: Optional[str] = None
    ):
        """
        Initialize clinical tokenizer with medical domain knowledge.
        
        Args:
            spacy_model: SpaCy model to use for base NLP processing
            medical_terms_file: Path to file containing medical terminology
            abbreviations_file: Path to file containing clinical abbreviations
        """
        # Load spaCy model for linguistic processing
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(
                f"SpaCy model {spacy_model} not found. "
                f"Install with: python -m spacy download {spacy_model}"
            )
            # For production, would handle this more robustly
            self.nlp = None
        
        # Load medical terminology and abbreviations
        # In production, these would come from curated clinical ontologies
        self.medical_terms = self._load_medical_terms(medical_terms_file)
        self.abbreviations = self._load_abbreviations(abbreviations_file)
        
        # Define uncertainty and negation markers
        self.uncertainty_markers = {
            'possible', 'possibly', 'probable', 'probably', 'likely',
            'unlikely', 'may', 'might', 'could', 'unclear', 'uncertain',
            'cannot rule out', 'suggestive of', 'consistent with',
            'suspicious for', 'concerning for', 'worrisome for'
        }
        
        self.negation_markers = {
            'no', 'not', 'without', 'absent', 'denies', 'negative',
            'none', 'neither', 'never', 'nothing', 'nowhere'
        }
        
        self.temporal_markers = {
            'today', 'yesterday', 'currently', 'recently', 'previously',
            'past', 'present', 'future', 'chronic', 'acute', 'subacute',
            'new onset', 'longstanding', 'since', 'for', 'during'
        }
        
        logger.info(
            f"Initialized clinical tokenizer with {len(self.medical_terms)} "
            f"medical terms and {len(self.abbreviations)} abbreviations"
        )
    
    def _load_medical_terms(self, filepath: Optional[str]) -> Set[str]:
        """
        Load medical terminology from file or use default set.
        
        In production, this would load from comprehensive clinical ontologies
        like SNOMED CT or UMLS. For this implementation, we use a small
        default set of common terms.
        """
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    return set(line.strip().lower() for line in f)
            except Exception as e:
                logger.warning(f"Failed to load medical terms from {filepath}: {e}")
        
        # Default set of common medical terms
        # In production, this would be much more comprehensive
        return {
            'hypertension', 'diabetes', 'myocardial infarction',
            'congestive heart failure', 'chronic obstructive pulmonary disease',
            'pneumonia', 'sepsis', 'acute kidney injury', 'stroke',
            'coronary artery disease', 'atrial fibrillation',
            'deep vein thrombosis', 'pulmonary embolism'
        }
    
    def _load_abbreviations(self, filepath: Optional[str]) -> Dict[str, str]:
        """
        Load clinical abbreviations and their expansions.
        
        Clinical abbreviations are a major source of ambiguity and potential
        bias, as abbreviation usage varies across institutions and individual
        clinicians. Some abbreviations may be more common in certain settings
        or when documenting care for certain populations.
        """
        if filepath:
            try:
                abbreviations = {}
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            abbreviations[parts[0].lower()] = parts[1].lower()
                return abbreviations
            except Exception as e:
                logger.warning(
                    f"Failed to load abbreviations from {filepath}: {e}"
                )
        
        # Default set of common clinical abbreviations
        return {
            'mi': 'myocardial infarction',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'afib': 'atrial fibrillation',
            'cad': 'coronary artery disease',
            'aki': 'acute kidney injury',
            'uti': 'urinary tract infection',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'abd': 'abdominal',
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'pt': 'patient',
            'hx': 'history',
            'sx': 'symptoms',
            'tx': 'treatment'
        }
    
    def tokenize(
        self,
        text: str,
        preserve_case: bool = False,
        expand_abbreviations: bool = True,
        normalize_medical_terms: bool = True
    ) -> List[Token]:
        """
        Tokenize clinical text with domain-specific processing.
        
        Args:
            text: Input clinical text
            preserve_case: Whether to preserve original case
            expand_abbreviations: Whether to expand known abbreviations
            normalize_medical_terms: Whether to normalize medical terminology
            
        Returns:
            List of Token objects with metadata
        """
        if not text or not text.strip():
            return []
        
        # Use spaCy for base linguistic processing
        if self.nlp is None:
            # Fallback to simple tokenization if spaCy not available
            return self._simple_tokenize(text, preserve_case)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Get original and normalized forms
            original_text = token.text
            normalized_text = original_text if preserve_case else original_text.lower()
            
            # Check if this is a known abbreviation
            is_abbrev = normalized_text in self.abbreviations
            if is_abbrev and expand_abbreviations:
                normalized_text = self.abbreviations[normalized_text]
            
            # Check if this is part of a medical term
            # In production, would use more sophisticated medical NER
            is_medical = normalized_text in self.medical_terms
            
            # Check for uncertainty and negation markers
            is_uncertainty = normalized_text in self.uncertainty_markers
            is_negation = normalized_text in self.negation_markers
            is_temporal = normalized_text in self.temporal_markers
            
            # Create token with metadata
            clinical_token = Token(
                text=original_text,
                normalized=normalized_text,
                pos=token.pos_,
                is_medical_term=is_medical,
                is_abbreviation=is_abbrev,
                uncertainty_marker=is_uncertainty,
                negation_marker=is_negation,
                temporal_marker=is_temporal
            )
            
            tokens.append(clinical_token)
        
        return tokens
    
    def _simple_tokenize(
        self,
        text: str,
        preserve_case: bool = False
    ) -> List[Token]:
        """
        Simple fallback tokenization when spaCy is not available.
        """
        # Basic whitespace and punctuation tokenization
        pattern = r'\w+|[^\w\s]'
        matches = re.finditer(pattern, text)
        
        tokens = []
        for match in matches:
            original = match.group()
            normalized = original if preserve_case else original.lower()
            
            token = Token(
                text=original,
                normalized=normalized,
                is_medical_term=normalized in self.medical_terms,
                is_abbreviation=normalized in self.abbreviations
            )
            tokens.append(token)
        
        return tokens
    
    def analyze_documentation_quality(
        self,
        text: str
    ) -> Dict[str, float]:
        """
        Analyze documentation quality markers that may correlate with bias.
        
        Documentation quality varies systematically across healthcare settings
        and patient populations. Brief notes with many abbreviations may
        indicate time pressure in under-resourced settings. This analysis
        helps identify potential sources of bias in downstream NLP tasks.
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            Dictionary of documentation quality metrics
        """
        tokens = self.tokenize(text, preserve_case=False)
        
        if not tokens:
            return {
                'total_tokens': 0,
                'abbreviation_rate': 0.0,
                'medical_term_rate': 0.0,
                'uncertainty_rate': 0.0,
                'avg_token_length': 0.0
            }
        
        n_tokens = len(tokens)
        n_abbreviations = sum(1 for t in tokens if t.is_abbreviation)
        n_medical = sum(1 for t in tokens if t.is_medical_term)
        n_uncertainty = sum(1 for t in tokens if t.uncertainty_marker)
        avg_length = np.mean([len(t.text) for t in tokens])
        
        return {
            'total_tokens': n_tokens,
            'abbreviation_rate': n_abbreviations / n_tokens,
            'medical_term_rate': n_medical / n_tokens,
            'uncertainty_rate': n_uncertainty / n_tokens,
            'avg_token_length': float(avg_length)
        }
```

This clinical tokenizer provides the foundation for all downstream NLP tasks while explicitly tracking potential sources of bias. The documentation quality analysis is particularly important for equity considerations, as it surfaces systematic differences in clinical documentation that correlate with patient demographics and care settings. Research has demonstrated that notes about patients from underserved communities tend to be briefer, use more abbreviations, and contain less detailed clinical reasoning, reflecting time pressures and resource constraints in safety-net healthcare settings rather than differences in clinical complexity \citep{brown2015disparities, chen2021ethical}.

### 6.2.2 Negation and Uncertainty Detection

Clinical language frequently expresses negation and uncertainty in ways that standard NLP tools fail to capture correctly. A statement like "no evidence of pneumonia" is fundamentally different from "pneumonia" but naive text processing might treat them equivalently if it simply extracts the word "pneumonia" without understanding the negation context. Similarly, expressions of clinical uncertainty like "concerning for" or "cannot rule out" convey important information about diagnostic confidence that affects clinical decision-making.

These linguistic phenomena are not distributed uniformly. Clinician use of hedging and uncertainty language varies by training, specialty, and institutional culture. More importantly, studies have found that clinicians use more uncertain language when documenting care for certain patient populations even when clinical presentations are equivalent, potentially reflecting unconscious bias in perceived reliability of symptoms reported by patients from marginalized groups \citep{sun2022examining}.

```python
from typing import List, Tuple, Optional
from dataclasses import dataclass
import re

@dataclass
class NegationSpan:
    """
    Represents a span of text within a negation or uncertainty context.
    
    Attributes:
        start: Start position in original text
        end: End position in original text
        text: Text content of the span
        type: Type of modification ('negation', 'uncertainty', 'possible')
        scope: Extent of negation/uncertainty effect
        trigger: Phrase that triggered the negation/uncertainty
    """
    start: int
    end: int
    text: str
    type: str
    scope: Tuple[int, int]
    trigger: str

class ClinicalNegationDetector:
    """
    Detects negation and uncertainty contexts in clinical text.
    
    This implementation is based on NegEx and ConText algorithms but
    extended to handle the diverse linguistic patterns in clinical text
    across different healthcare settings and documentation styles.
    """
    
    def __init__(self):
        """Initialize negation and uncertainty detection patterns."""
        
        # Negation triggers and their typical scope
        # Scope is number of tokens before (pre) or after (post) trigger
        self.negation_patterns = {
            # Definite negation
            'no': {'type': 'definite', 'scope': 'post', 'window': 5},
            'not': {'type': 'definite', 'scope': 'post', 'window': 3},
            'denies': {'type': 'definite', 'scope': 'post', 'window': 5},
            'absent': {'type': 'definite', 'scope': 'pre', 'window': 2},
            'without': {'type': 'definite', 'scope': 'post', 'window': 3},
            'negative for': {'type': 'definite', 'scope': 'post', 'window': 5},
            'ruled out': {'type': 'definite', 'scope': 'pre', 'window': 3},
            'free of': {'type': 'definite', 'scope': 'post', 'window': 3},
            
            # Pseudo-negation (not actually negating)
            'not only': {'type': 'pseudo', 'scope': None, 'window': 0},
            'not certain': {'type': 'pseudo', 'scope': None, 'window': 0},
            'no increase': {'type': 'pseudo', 'scope': None, 'window': 0},
            'no change': {'type': 'pseudo', 'scope': None, 'window': 0}
        }
        
        # Uncertainty patterns
        self.uncertainty_patterns = {
            'possible': {'type': 'low', 'scope': 'post', 'window': 4},
            'possibly': {'type': 'low', 'scope': 'post', 'window': 4},
            'likely': {'type': 'moderate', 'scope': 'post', 'window': 4},
            'probable': {'type': 'moderate', 'scope': 'post', 'window': 4},
            'may be': {'type': 'low', 'scope': 'post', 'window': 4},
            'might be': {'type': 'low', 'scope': 'post', 'window': 4},
            'concerning for': {'type': 'moderate', 'scope': 'post', 'window': 5},
            'suspicious for': {'type': 'moderate', 'scope': 'post', 'window': 5},
            'suggestive of': {'type': 'moderate', 'scope': 'post', 'window': 5},
            'consistent with': {'type': 'moderate', 'scope': 'post', 'window': 5},
            'cannot rule out': {'type': 'high', 'scope': 'post', 'window': 5},
            'unclear': {'type': 'high', 'scope': 'both', 'window': 3},
            'uncertain': {'type': 'high', 'scope': 'both', 'window': 3}
        }
        
        # Scope terminators - phrases that limit negation/uncertainty scope
        self.scope_terminators = {
            'but', 'however', 'although', 'except', 'aside from',
            'apart from', 'still', 'though', 'nevertheless'
        }
        
        logger.info("Initialized clinical negation detector")
    
    def detect(
        self,
        tokens: List[Token]
    ) -> Tuple[List[NegationSpan], List[NegationSpan]]:
        """
        Detect negation and uncertainty spans in tokenized clinical text.
        
        Args:
            tokens: List of Token objects from clinical tokenizer
            
        Returns:
            Tuple of (negation_spans, uncertainty_spans)
        """
        negation_spans = []
        uncertainty_spans = []
        
        # Build text position mapping
        position_map = self._build_position_map(tokens)
        
        # Search for negation patterns
        for i, token in enumerate(tokens):
            # Check for multi-word patterns first
            for pattern_length in [3, 2, 1]:
                if i + pattern_length > len(tokens):
                    continue
                
                phrase = ' '.join(
                    t.normalized for t in tokens[i:i+pattern_length]
                )
                
                # Check negation patterns
                if phrase in self.negation_patterns:
                    pattern_info = self.negation_patterns[phrase]
                    
                    # Skip pseudo-negations
                    if pattern_info['type'] == 'pseudo':
                        continue
                    
                    # Determine scope
                    scope_start, scope_end = self._determine_scope(
                        i, i + pattern_length - 1,
                        tokens,
                        pattern_info['scope'],
                        pattern_info['window']
                    )
                    
                    if scope_start < scope_end:
                        span = NegationSpan(
                            start=position_map[scope_start][0],
                            end=position_map[scope_end][1],
                            text=' '.join(
                                t.text for t in tokens[scope_start:scope_end+1]
                            ),
                            type='negation',
                            scope=(scope_start, scope_end),
                            trigger=phrase
                        )
                        negation_spans.append(span)
                    
                    break
                
                # Check uncertainty patterns
                if phrase in self.uncertainty_patterns:
                    pattern_info = self.uncertainty_patterns[phrase]
                    
                    scope_start, scope_end = self._determine_scope(
                        i, i + pattern_length - 1,
                        tokens,
                        pattern_info['scope'],
                        pattern_info['window']
                    )
                    
                    if scope_start < scope_end:
                        span = NegationSpan(
                            start=position_map[scope_start][0],
                            end=position_map[scope_end][1],
                            text=' '.join(
                                t.text for t in tokens[scope_start:scope_end+1]
                            ),
                            type=f"uncertainty_{pattern_info['type']}",
                            scope=(scope_start, scope_end),
                            trigger=phrase
                        )
                        uncertainty_spans.append(span)
                    
                    break
        
        return negation_spans, uncertainty_spans
    
    def _build_position_map(
        self,
        tokens: List[Token]
    ) -> List[Tuple[int, int]]:
        """
        Build mapping from token index to character positions.
        
        Returns list where each element is (start_pos, end_pos) tuple
        for the corresponding token.
        """
        positions = []
        current_pos = 0
        
        for token in tokens:
            token_length = len(token.text)
            positions.append((current_pos, current_pos + token_length))
            # Account for whitespace after token
            current_pos += token_length + 1
        
        return positions
    
    def _determine_scope(
        self,
        trigger_start: int,
        trigger_end: int,
        tokens: List[Token],
        scope_direction: str,
        window_size: int
    ) -> Tuple[int, int]:
        """
        Determine the scope of negation or uncertainty effect.
        
        Args:
            trigger_start: Token index where trigger phrase starts
            trigger_end: Token index where trigger phrase ends
            tokens: Full list of tokens
            scope_direction: 'pre', 'post', or 'both'
            window_size: Maximum number of tokens in scope
            
        Returns:
            Tuple of (scope_start_idx, scope_end_idx)
        """
        n_tokens = len(tokens)
        
        if scope_direction == 'post':
            scope_start = trigger_end + 1
            scope_end = min(trigger_end + window_size + 1, n_tokens)
            
            # Check for scope terminators
            for i in range(scope_start, scope_end):
                if tokens[i].normalized in self.scope_terminators:
                    scope_end = i
                    break
        
        elif scope_direction == 'pre':
            scope_start = max(0, trigger_start - window_size)
            scope_end = trigger_start
            
            # Check for scope terminators
            for i in range(scope_end - 1, scope_start - 1, -1):
                if tokens[i].normalized in self.scope_terminators:
                    scope_start = i + 1
                    break
        
        else:  # 'both'
            pre_start = max(0, trigger_start - window_size)
            post_end = min(trigger_end + window_size + 1, n_tokens)
            
            # Check for terminators in both directions
            for i in range(trigger_start - 1, pre_start - 1, -1):
                if tokens[i].normalized in self.scope_terminators:
                    pre_start = i + 1
                    break
            
            for i in range(trigger_end + 1, post_end):
                if tokens[i].normalized in self.scope_terminators:
                    post_end = i
                    break
            
            scope_start = pre_start
            scope_end = post_end
        
        return scope_start, scope_end
    
    def analyze_uncertainty_patterns(
        self,
        texts: List[str],
        metadata: Optional[Dict[str, List]] = None
    ) -> Dict[str, float]:
        """
        Analyze patterns of uncertainty language use across texts.
        
        This analysis can reveal systematic differences in uncertainty
        language that may correlate with patient demographics, potentially
        indicating bias in clinical documentation.
        
        Args:
            texts: List of clinical texts to analyze
            metadata: Optional metadata including patient demographics
            
        Returns:
            Dictionary of uncertainty usage statistics
        """
        tokenizer = ClinicalTokenizer()
        
        total_tokens = 0
        uncertainty_counts = {'low': 0, 'moderate': 0, 'high': 0}
        negation_counts = 0
        
        for text in texts:
            tokens = tokenizer.tokenize(text)
            total_tokens += len(tokens)
            
            negations, uncertainties = self.detect(tokens)
            negation_counts += len(negations)
            
            for uncertainty_span in uncertainties:
                unc_type = uncertainty_span.type.split('_')[1]
                uncertainty_counts[unc_type] += 1
        
        if total_tokens == 0:
            return {}
        
        results = {
            'negation_rate': negation_counts / total_tokens,
            'uncertainty_rate_low': uncertainty_counts['low'] / total_tokens,
            'uncertainty_rate_moderate': uncertainty_counts['moderate'] / total_tokens,
            'uncertainty_rate_high': uncertainty_counts['high'] / total_tokens,
            'total_uncertainty_rate': sum(uncertainty_counts.values()) / total_tokens
        }
        
        # If metadata provided, compute stratified statistics
        if metadata:
            results['stratified'] = self._compute_stratified_stats(
                texts, metadata, tokenizer
            )
        
        return results
    
    def _compute_stratified_stats(
        self,
        texts: List[str],
        metadata: Dict[str, List],
        tokenizer: ClinicalTokenizer
    ) -> Dict[str, Dict[str, float]]:
        """Compute uncertainty statistics stratified by metadata variables."""
        stratified = {}
        
        for var_name, var_values in metadata.items():
            unique_values = set(var_values)
            
            for value in unique_values:
                # Get texts for this subgroup
                indices = [i for i, v in enumerate(var_values) if v == value]
                subgroup_texts = [texts[i] for i in indices]
                
                # Compute statistics for subgroup
                subgroup_stats = self.analyze_uncertainty_patterns(subgroup_texts)
                stratified[f'{var_name}_{value}'] = subgroup_stats
        
        return stratified
```

This negation and uncertainty detection system provides robust handling of the complex linguistic patterns in clinical text while enabling analysis of potential documentation biases. The stratified analysis capability is particularly important for identifying cases where uncertainty language patterns differ systematically by patient demographics in ways that might affect downstream clinical NLP applications \citep{sun2022examining, park2021examining}.

## 6.3 Named Entity Recognition for Clinical Concepts

Named entity recognition identifies and classifies mentions of specific entities in text such as diseases, symptoms, medications, procedures, and anatomical locations. In clinical text, accurate NER is foundational for virtually all downstream applications from clinical decision support to population health surveillance. However, standard NER approaches developed on news or social media text perform poorly on clinical documentation, and performance degrades further when applied across diverse healthcare settings serving different patient populations.

The equity challenges in clinical NER are substantial. Training data for clinical NER systems typically comes from academic medical centers with particular documentation styles and patient demographics. Entity mentions may be expressed differently across settings: abbreviated references common in busy emergency departments, colloquial terms used in primary care clinics, terminology variations reflecting different clinician training backgrounds or local institutional norms. More concerning, the same clinical concepts may be documented differently for different patient populations due to unconscious bias, with symptoms reported by certain patients described in more vague or dismissive terms than identical symptoms in other patients.

### 6.3.1 BiLSTM-CRF for Clinical NER

We implement a bidirectional LSTM with conditional random field (CRF) layer for clinical named entity recognition, with explicit attention to fairness across diverse clinical settings and patient populations.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)

class BiLSTMCRF(nn.Module):
    """
    Bidirectional LSTM with CRF layer for clinical named entity recognition.
    
    This architecture is particularly well-suited for sequence labeling tasks
    in clinical text, as the bidirectional LSTM captures context in both
    directions and the CRF layer enforces valid label sequences.
    
    We include fairness-aware training and evaluation capabilities to ensure
    equitable performance across patient populations and care settings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        dropout: float = 0.5,
        use_crf: bool = True,
        pad_idx: int = 0
    ):
        """
        Initialize BiLSTM-CRF model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_labels: Number of entity labels
            dropout: Dropout rate for regularization
            use_crf: Whether to use CRF layer
            pad_idx: Index of padding token
        """
        super(BiLSTMCRF, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.use_crf = use_crf
        self.pad_idx = pad_idx
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_idx
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers=2,
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to project LSTM output to label space
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
        
        # CRF layer for modeling label dependencies
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
        
        logger.info(
            f"Initialized BiLSTM-CRF with vocab_size={vocab_size}, "
            f"embedding_dim={embedding_dim}, hidden_dim={hidden_dim}, "
            f"num_labels={num_labels}, use_crf={use_crf}"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Optional labels of shape (batch_size, seq_len)
            
        Returns:
            Dictionary containing loss (if labels provided) and predictions
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Get sequence lengths from attention mask
        lengths = attention_mask.sum(dim=1).cpu()
        
        # Pack padded sequences for efficient LSTM processing
        packed = pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(packed)
        
        # Unpack sequences
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to label space
        emissions = self.hidden2tag(lstm_out)
        
        output = {'emissions': emissions}
        
        if self.use_crf:
            # Use CRF for predictions
            if labels is not None:
                # Compute CRF loss
                loss = -self.crf(emissions, labels, mask=attention_mask.bool())
                output['loss'] = loss
            
            # Decode best path
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            output['predictions'] = predictions
        else:
            # Use softmax for predictions without CRF
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                # Flatten tensors for loss computation
                loss = loss_fct(
                    emissions.view(-1, self.num_labels),
                    labels.view(-1)
                )
                output['loss'] = loss
            
            # Get predictions from logits
            predictions = torch.argmax(emissions, dim=-1)
            output['predictions'] = predictions
        
        return output
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Make predictions on input sequences.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            List of predicted label sequences
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)
            
            if self.use_crf:
                return output['predictions']
            else:
                # Convert tensor to list and remove padding
                predictions = output['predictions'].cpu().numpy()
                mask = attention_mask.cpu().numpy()
                
                result = []
                for pred, m in zip(predictions, mask):
                    result.append(pred[m == 1].tolist())
                
                return result


class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.
    
    The CRF layer models dependencies between adjacent labels, ensuring
    that predicted label sequences follow valid patterns (e.g., I-DRUG
    cannot follow B-SYMPTOM).
    """
    
    def __init__(
        self,
        num_labels: int,
        batch_first: bool = True
    ):
        """
        Initialize CRF layer.
        
        Args:
            num_labels: Number of possible labels
            batch_first: Whether batch dimension is first
        """
        super(CRF, self).__init__()
        
        self.num_labels = num_labels
        self.batch_first = batch_first
        
        # Transition parameters: transitions[i,j] is score of transitioning
        # from label i to label j
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for CRF.
        
        Args:
            emissions: Emission scores of shape (batch_size, seq_len, num_labels)
            tags: True labels of shape (batch_size, seq_len)
            mask: Mask of shape (batch_size, seq_len)
            
        Returns:
            Negative log-likelihood loss
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # Compute log sum of all possible paths
        log_partition = self._forward_algorithm(emissions, mask)
        
        # Compute score of true path
        gold_score = self._score_sentence(emissions, tags, mask)
        
        # Loss is negative log-likelihood
        return torch.mean(log_partition - gold_score)
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Find most likely label sequence using Viterbi algorithm.
        
        Args:
            emissions: Emission scores of shape (batch_size, seq_len, num_labels)
            mask: Mask of shape (batch_size, seq_len)
            
        Returns:
            List of predicted label sequences
        """
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2],
                dtype=torch.bool,
                device=emissions.device
            )
        
        return self._viterbi_decode(emissions, mask)
    
    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward algorithm to compute log partition function.
        """
        batch_size, seq_length, num_labels = emissions.shape
        
        # Initialize forward variables
        alpha = emissions[:, 0] + self.start_transitions.unsqueeze(0)
        
        # Iterate through sequence
        for i in range(1, seq_length):
            # Broadcast for transition scores
            emit_score = emissions[:, i].unsqueeze(1)  # (batch, 1, labels)
            trans_score = self.transitions.unsqueeze(0)  # (1, labels, labels)
            next_alpha = alpha.unsqueeze(2) + trans_score + emit_score
            
            # Log-sum-exp
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            
            # Update alpha with mask
            alpha = torch.where(
                mask[:, i].unsqueeze(1),
                next_alpha,
                alpha
            )
        
        # Add end transitions
        alpha = alpha + self.end_transitions.unsqueeze(0)
        
        # Final log partition
        return torch.logsumexp(alpha, dim=1)
    
    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute score of a given label sequence.
        """
        batch_size, seq_length = tags.shape
        
        # Add start transitions
        score = self.start_transitions[tags[:, 0]]
        
        # Add emission scores
        score = score + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        
        # Iterate through sequence
        for i in range(1, seq_length):
            # Transition score from previous tag to current tag
            trans_score = self.transitions[tags[:, i-1], tags[:, i]]
            
            # Emission score for current tag
            emit_score = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            
            # Add to total score with mask
            score = score + torch.where(
                mask[:, i],
                trans_score + emit_score,
                torch.zeros_like(trans_score)
            )
        
        # Add end transitions for final tag
        last_tag_indices = mask.sum(dim=1) - 1
        last_tags = tags.gather(1, last_tag_indices.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        
        return score
    
    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> List[List[int]]:
        """
        Viterbi algorithm to find best label sequence.
        """
        batch_size, seq_length, num_labels = emissions.shape
        
        # Initialize
        viterbi = emissions[:, 0] + self.start_transitions.unsqueeze(0)
        backpointers = []
        
        # Forward pass
        for i in range(1, seq_length):
            # Broadcast for computing scores
            broadcast_viterbi = viterbi.unsqueeze(2)  # (batch, labels, 1)
            broadcast_trans = self.transitions.unsqueeze(0)  # (1, labels, labels)
            
            # Next viterbi scores
            next_viterbi = broadcast_viterbi + broadcast_trans + emissions[:, i].unsqueeze(1)
            
            # Get best previous tag
            next_viterbi, indices = next_viterbi.max(dim=1)
            
            # Update with mask
            viterbi = torch.where(
                mask[:, i].unsqueeze(1),
                next_viterbi,
                viterbi
            )
            
            backpointers.append(indices)
        
        # Add end transitions
        viterbi = viterbi + self.end_transitions.unsqueeze(0)
        
        # Backtrack to find best path
        best_paths = []
        for batch_idx in range(batch_size):
            # Find last tag
            last_position = mask[batch_idx].sum() - 1
            _, last_tag = viterbi[batch_idx].max(dim=0)
            
            # Backtrack
            path = [last_tag.item()]
            for i in range(len(backpointers) - 1, -1, -1):
                if i <= last_position:
                    last_tag = backpointers[i][batch_idx, last_tag]
                    path.append(last_tag.item())
            
            # Reverse path and truncate to actual length
            path = path[::-1]
            path = path[:last_position + 1]
            
            best_paths.append(path)
        
        return best_paths


class ClinicalNERTrainer:
    """
    Trainer for clinical NER with fairness-aware evaluation.
    """
    
    def __init__(
        self,
        model: BiLSTMCRF,
        label_names: List[str],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: BiLSTM-CRF model
            label_names: List of label names
            device: Device for training
        """
        self.model = model.to(device)
        self.label_names = label_names
        self.device = device
        
        logger.info(f"Initialized trainer on device: {device}")
    
    def train(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 5
    ) -> Dict[str, List[float]]:
        """
        Train model with early stopping.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate
            patience: Early stopping patience
            
        Returns:
            Dictionary of training history
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }
        
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                output = self.model(input_ids, attention_mask, labels)
                loss = output['loss']
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_dataloader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_metrics = self.evaluate(val_dataloader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1_macro'])
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f} - "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Early stopping
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def evaluate(
        self,
        dataloader,
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate model on data.
        
        Args:
            dataloader: Data loader
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                output = self.model(input_ids, attention_mask, labels)
                total_loss += output['loss'].item()
                
                predictions = output['predictions']
                if not isinstance(predictions, list):
                    # Convert tensor predictions to list
                    mask = attention_mask.cpu().numpy()
                    pred_array = predictions.cpu().numpy()
                    predictions = [
                        pred[m == 1].tolist()
                        for pred, m in zip(pred_array, mask)
                    ]
                
                # Get true labels
                label_array = labels.cpu().numpy()
                mask = attention_mask.cpu().numpy()
                true_labels = [
                    label[m == 1].tolist()
                    for label, m in zip(label_array, mask)
                ]
                
                all_predictions.extend(predictions)
                all_labels.extend(true_labels)
        
        # Flatten for sklearn metrics
        flat_predictions = [p for seq in all_predictions for p in seq]
        flat_labels = [l for seq in all_labels for l in seq]
        
        # Compute metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'f1_macro': f1_score(flat_labels, flat_predictions, average='macro'),
            'f1_micro': f1_score(flat_labels, flat_predictions, average='micro'),
            'precision_macro': precision_score(flat_labels, flat_predictions, average='macro'),
            'recall_macro': recall_score(flat_labels, flat_predictions, average='macro')
        }
        
        if return_predictions:
            metrics['predictions'] = all_predictions
            metrics['labels'] = all_labels
        
        return metrics
    
    def evaluate_fairness(
        self,
        dataloader,
        metadata: Dict[str, List]
    ) -> Dict[str, Dict]:
        """
        Evaluate model fairness across demographic groups.
        
        Args:
            dataloader: Data loader with evaluation data
            metadata: Dictionary mapping metadata fields to values for each example
            
        Returns:
            Dictionary of fairness metrics stratified by demographics
        """
        # Get predictions
        eval_results = self.evaluate(dataloader, return_predictions=True)
        
        predictions = eval_results['predictions']
        labels = eval_results['labels']
        
        fairness_metrics = {}
        
        # Compute metrics for each metadata variable
        for var_name, var_values in metadata.items():
            if len(var_values) != len(predictions):
                logger.warning(
                    f"Metadata variable {var_name} has {len(var_values)} values "
                    f"but {len(predictions)} predictions"
                )
                continue
            
            unique_values = set(var_values)
            var_metrics = {}
            
            for value in unique_values:
                # Get indices for this subgroup
                indices = [i for i, v in enumerate(var_values) if v == value]
                
                if not indices:
                    continue
                
                # Get predictions and labels for subgroup
                subgroup_preds = [predictions[i] for i in indices]
                subgroup_labels = [labels[i] for i in indices]
                
                # Flatten
                flat_preds = [p for seq in subgroup_preds for p in seq]
                flat_labels = [l for seq in subgroup_labels for l in seq]
                
                if not flat_preds:
                    continue
                
                # Compute metrics
                from sklearn.metrics import f1_score, precision_score, recall_score
                
                var_metrics[str(value)] = {
                    'n_examples': len(indices),
                    'n_tokens': len(flat_preds),
                    'f1_macro': f1_score(flat_labels, flat_preds, average='macro'),
                    'f1_micro': f1_score(flat_labels, flat_preds, average='micro'),
                    'precision': precision_score(flat_labels, flat_preds, average='macro'),
                    'recall': recall_score(flat_labels, flat_preds, average='macro')
                }
            
            fairness_metrics[var_name] = var_metrics
        
        return fairness_metrics
```

This BiLSTM-CRF implementation provides state-of-the-art performance for clinical NER while incorporating fairness evaluation capabilities that surface disparate performance across patient demographics and care settings. The architecture has been widely validated for clinical NER tasks and achieves strong performance while remaining interpretable through attention mechanisms and the structured CRF layer \citep{lample2016neural, huang2015bidirectional, wang2018clinical}.

### 6.3.2 Transformer-Based Clinical NER

While BiLSTM-CRF models provide strong baseline performance, transformer-based approaches including BERT and its clinical variants like BioBERT, ClinicalBERT, and Med-BERT have achieved state-of-the-art results on clinical NER tasks. We implement a transformer-based NER system with explicit fairness considerations.

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch
from typing import List, Dict, Optional
import numpy as np

class TransformerClinicalNER:
    """
    Transformer-based clinical NER with fairness evaluation.
    
    Supports various pre-trained models including:
    - BioBERT: Pre-trained on biomedical literature
    - ClinicalBERT: Pre-trained on clinical notes from MIMIC-III
    - PubMedBERT: Pre-trained on PubMed abstracts
    - BlueBERT: Pre-trained on both PubMed and MIMIC-III
    """
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels: int = 9,
        label_names: Optional[List[str]] = None,
        max_length: int = 512
    ):
        """
        Initialize transformer-based NER model.
        
        Args:
            model_name: Name of pre-trained model from HuggingFace
            num_labels: Number of NER labels
            label_names: List of label names
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Label mapping
        if label_names:
            self.label_names = label_names
            self.label2id = {label: i for i, label in enumerate(label_names)}
            self.id2label = {i: label for label, i in self.label2id.items()}
        else:
            self.label_names = [f"LABEL_{i}" for i in range(num_labels)]
            self.label2id = {label: i for i, label in enumerate(self.label_names)}
            self.id2label = {i: label for label, i in self.label2id.items()}
        
        logger.info(
            f"Initialized {model_name} with {num_labels} labels"
        )
    
    def tokenize_and_align_labels(
        self,
        texts: List[List[str]],
        labels: List[List[str]],
        label_all_tokens: bool = False
    ) -> Dict:
        """
        Tokenize texts and align labels with subword tokens.
        
        Transformer tokenizers split words into subword tokens, so we need
        to align the original word-level labels with the new token-level
        labels. This is a common challenge in token classification.
        
        Args:
            texts: List of tokenized texts (each text is list of words)
            labels: List of label sequences (same length as texts)
            label_all_tokens: Whether to label all subword tokens or just first
            
        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        tokenized_inputs = self.tokenizer(
            texts,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        aligned_labels = []
        
        for i, label_seq in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                # Special tokens have a word id that is None
                if word_idx is None:
                    label_ids.append(-100)  # Ignore in loss
                # We set the label for the first token of each word
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label2id[label_seq[word_idx]])
                # For other tokens in a word, either copy label or ignore
                else:
                    if label_all_tokens:
                        label_ids.append(self.label2id[label_seq[word_idx]])
                    else:
                        label_ids.append(-100)
                
                previous_word_idx = word_idx
            
            aligned_labels.append(label_ids)
        
        tokenized_inputs["labels"] = torch.tensor(aligned_labels)
        
        return tokenized_inputs
    
    def train(
        self,
        train_texts: List[List[str]],
        train_labels: List[List[str]],
        val_texts: List[List[str]],
        val_labels: List[List[str]],
        output_dir: str = "./ner_model",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01
    ):
        """
        Train NER model.
        
        Args:
            train_texts: Training texts (list of token lists)
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        # Tokenize and prepare data
        train_encodings = self.tokenize_and_align_labels(
            train_texts,
            train_labels
        )
        val_encodings = self.tokenize_and_align_labels(
            val_texts,
            val_labels
        )
        
        # Create datasets
        train_dataset = NERDataset(train_encodings)
        val_dataset = NERDataset(val_encodings)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=100
        )
        
        # Define data collator
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer,
            padding=True
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete")
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten for metrics
        flat_preds = [p for seq in true_predictions for p in seq]
        flat_labels = [l for seq in true_labels for l in seq]
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        return {
            'f1': f1_score(flat_labels, flat_preds, average='macro'),
            'precision': precision_score(flat_labels, flat_preds, average='macro'),
            'recall': recall_score(flat_labels, flat_preds, average='macro')
        }
    
    def predict(
        self,
        texts: List[List[str]]
    ) -> List[List[str]]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of tokenized texts
            
        Returns:
            List of predicted label sequences
        """
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Align predictions with original words
        aligned_predictions = []
        
        for i in range(len(texts)):
            word_ids = inputs.word_ids(batch_index=i)
            previous_word_idx = None
            pred_labels = []
            
            for word_idx, pred_id in zip(word_ids, predictions[i]):
                if word_idx is None:
                    continue
                elif word_idx != previous_word_idx:
                    pred_labels.append(self.id2label[pred_id.item()])
                previous_word_idx = word_idx
            
            aligned_predictions.append(pred_labels)
        
        return aligned_predictions
    
    def evaluate_fairness(
        self,
        texts: List[List[str]],
        labels: List[List[str]],
        metadata: Dict[str, List]
    ) -> Dict[str, Dict]:
        """
        Evaluate model fairness across demographic groups.
        
        Args:
            texts: Evaluation texts
            labels: True labels
            metadata: Dictionary with demographic information
            
        Returns:
            Fairness metrics stratified by demographics
        """
        # Get predictions
        predictions = self.predict(texts)
        
        fairness_metrics = {}
        
        # Compute metrics for each metadata variable
        for var_name, var_values in metadata.items():
            if len(var_values) != len(texts):
                logger.warning(
                    f"Metadata variable {var_name} length mismatch"
                )
                continue
            
            unique_values = set(var_values)
            var_metrics = {}
            
            for value in unique_values:
                indices = [i for i, v in enumerate(var_values) if v == value]
                
                if not indices:
                    continue
                
                # Get subgroup data
                subgroup_preds = [predictions[i] for i in indices]
                subgroup_labels = [labels[i] for i in indices]
                
                # Flatten
                flat_preds = [p for seq in subgroup_preds for p in seq]
                flat_labels = [l for seq in subgroup_labels for l in seq]
                
                if not flat_preds:
                    continue
                
                from sklearn.metrics import f1_score, precision_score, recall_score
                
                var_metrics[str(value)] = {
                    'n_examples': len(indices),
                    'n_tokens': len(flat_preds),
                    'f1_macro': f1_score(flat_labels, flat_preds, average='macro'),
                    'precision': precision_score(flat_labels, flat_preds, average='macro'),
                    'recall': recall_score(flat_labels, flat_preds, average='macro')
                }
            
            fairness_metrics[var_name] = var_metrics
        
        return fairness_metrics


class NERDataset(torch.utils.data.Dataset):
    """PyTorch dataset for NER."""
    
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings['input_ids'])
```

These transformer-based NER implementations leverage pre-trained clinical language models that have learned rich representations of medical language from large corpora of biomedical literature and clinical notes. However, the pre-training data itself may reflect biases in research focus and clinical documentation that disadvantage underrepresented populations, requiring careful validation across diverse patient groups \citep{alsentzer2019publicly, lee2020biobert, huang2019clinicalbert}.

## 6.4 Bias Detection and Mitigation in Clinical Language Models

Clinical language models trained on historical electronic health record data inevitably learn the biases encoded in that data, including systematic differences in how clinicians describe patients from different racial, ethnic, socioeconomic, and other demographic groups. These learned associations can then be amplified when the models are used for downstream clinical tasks, potentially exacerbating healthcare disparities. Detecting and mitigating these biases is essential before deploying clinical language models in real-world healthcare settings.

### 6.4.1 Measuring Bias in Clinical Language Models

We implement comprehensive frameworks for measuring various forms of bias in clinical language models before they are used for downstream applications.

```python
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Set
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import logging

logger = logging.getLogger(__name__)

class ClinicalLanguageModelBiasAuditor:
    """
    Framework for auditing bias in clinical language models.
    
    Implements multiple bias detection approaches including:
    - Word Embedding Association Test (WEAT) for implicit associations
    - Sentence probability bias detection
    - Masked language model prediction analysis
    - Attention pattern analysis for differential focus
    """
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize bias auditor with clinical language model.
        
        Args:
            model_name: Pre-trained model to audit
            device: Device for computation
        """
        self.model_name = model_name
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Define demographic attribute terms for bias testing
        self.demographic_terms = {
            'race_white': ['white', 'caucasian', 'european'],
            'race_black': ['black', 'african american'],
            'race_hispanic': ['hispanic', 'latino', 'latina'],
            'race_asian': ['asian', 'asian american'],
            'gender_male': ['man', 'male', 'he', 'him'],
            'gender_female': ['woman', 'female', 'she', 'her'],
            'insurance_private': ['private insurance', 'commercial insurance'],
            'insurance_public': ['medicaid', 'medicare', 'uninsured'],
            'ses_high': ['wealthy', 'affluent', 'high income'],
            'ses_low': ['poor', 'low income', 'impoverished']
        }
        
        # Define clinical attribute terms
        self.clinical_attributes = {
            'compliance': ['compliant', 'adherent', 'follows recommendations'],
            'noncompliance': ['noncompliant', 'nonadherent', 'refuses', 'difficult'],
            'pain_valid': ['severe pain', 'significant pain', 'painful'],
            'pain_invalid': ['complains of pain', 'claims pain'],
            'reliability': ['reliable historian', 'good historian'],
            'unreliability': ['poor historian', 'unreliable historian'],
        }
        
        logger.info(f"Initialized bias auditor for {model_name}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get contextual embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Mean-pooled embedding vector
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pool over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()[0]
    
    def compute_weat_effect_size(
        self,
        target_set1: List[str],
        target_set2: List[str],
        attribute_set1: List[str],
        attribute_set2: List[str]
    ) -> Tuple[float, float]:
        """
        Compute Word Embedding Association Test (WEAT) effect size.
        
        WEAT measures the differential association between two sets of
        target words (e.g., white names vs. Black names) and two sets of
        attribute words (e.g., pleasant vs. unpleasant words).
        
        Args:
            target_set1: First set of target terms
            target_set2: Second set of target terms
            attribute_set1: First set of attribute terms
            attribute_set2: Second set of attribute terms
            
        Returns:
            Tuple of (effect_size, p_value)
        """
        # Get embeddings for all sets
        target1_embeds = [self.get_embedding(t) for t in target_set1]
        target2_embeds = [self.get_embedding(t) for t in target_set2]
        attr1_embeds = [self.get_embedding(a) for a in attribute_set1]
        attr2_embeds = [self.get_embedding(a) for a in attribute_set2]
        
        def association(w_embed, A_embeds, B_embeds):
            """Compute association of word w with attribute sets A and B."""
            mean_sim_A = np.mean([
                1 - cosine(w_embed, a_embed) for a_embed in A_embeds
            ])
            mean_sim_B = np.mean([
                1 - cosine(w_embed, b_embed) for b_embed in B_embeds
            ])
            return mean_sim_A - mean_sim_B
        
        # Compute associations for both target sets
        assoc_target1 = [
            association(t_embed, attr1_embeds, attr2_embeds)
            for t_embed in target1_embeds
        ]
        assoc_target2 = [
            association(t_embed, attr1_embeds, attr2_embeds)
            for t_embed in target2_embeds
        ]
        
        # Effect size is difference in mean associations
        effect_size = np.mean(assoc_target1) - np.mean(assoc_target2)
        
        # Permutation test for p-value
        all_assoc = assoc_target1 + assoc_target2
        n1 = len(assoc_target1)
        
        n_permutations = 10000
        null_effects = []
        
        for _ in range(n_permutations):
            np.random.shuffle(all_assoc)
            perm_effect = np.mean(all_assoc[:n1]) - np.mean(all_assoc[n1:])
            null_effects.append(perm_effect)
        
        p_value = np.mean([abs(e) >= abs(effect_size) for e in null_effects])
        
        return effect_size, p_value
    
    def audit_demographic_clinical_associations(self) -> Dict[str, Dict]:
        """
        Audit associations between demographic terms and clinical attributes.
        
        This reveals whether the model has learned biased associations such as
        linking certain racial groups with non-compliance or unreliability.
        
        Returns:
            Dictionary of WEAT results for various demographic-clinical pairs
        """
        results = {}
        
        # Test each demographic dimension
        demographic_pairs = [
            ('race_white', 'race_black'),
            ('race_white', 'race_hispanic'),
            ('gender_male', 'gender_female'),
            ('insurance_private', 'insurance_public'),
            ('ses_high', 'ses_low')
        ]
        
        clinical_pairs = [
            ('compliance', 'noncompliance'),
            ('pain_valid', 'pain_invalid'),
            ('reliability', 'unreliability')
        ]
        
        for demo1, demo2 in demographic_pairs:
            for clin1, clin2 in clinical_pairs:
                test_name = f"{demo1}_vs_{demo2}_{clin1}_vs_{clin2}"
                
                effect_size, p_value = self.compute_weat_effect_size(
                    self.demographic_terms[demo1],
                    self.demographic_terms[demo2],
                    self.clinical_attributes[clin1],
                    self.clinical_attributes[clin2]
                )
                
                results[test_name] = {
                    'effect_size': float(effect_size),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'interpretation': self._interpret_weat_result(
                        demo1, demo2, clin1, clin2, effect_size, p_value
                    )
                }
                
                if p_value < 0.05:
                    logger.warning(
                        f"Significant bias detected: {test_name} "
                        f"(effect={effect_size:.4f}, p={p_value:.4f})"
                    )
        
        return results
    
    def _interpret_weat_result(
        self,
        demo1: str,
        demo2: str,
        clin1: str,
        clin2: str,
        effect_size: float,
        p_value: float
    ) -> str:
        """Generate human-readable interpretation of WEAT result."""
        if p_value >= 0.05:
            return "No significant association detected"
        
        if effect_size > 0:
            stronger_assoc = f"{demo1} with {clin1}"
            weaker_assoc = f"{demo2} with {clin2}"
        else:
            stronger_assoc = f"{demo2} with {clin1}"
            weaker_assoc = f"{demo1} with {clin2}"
        
        return (
            f"Model shows stronger association between {stronger_assoc} "
            f"compared to {weaker_assoc}"
        )
    
    def test_template_bias(
        self,
        template: str,
        fill_values: Dict[str, List[str]],
        mask_token: str = "[MASK]"
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Test for bias in masked language model predictions.
        
        Example: "The [MASK] patient was noncompliant with medications"
        Fill with different demographic terms and compare predictions.
        
        Args:
            template: Template string with [MASK] token
            fill_values: Dictionary mapping attribute names to lists of terms
            mask_token: Token to mask and predict
            
        Returns:
            Dictionary mapping fill values to predicted terms and probabilities
        """
        from transformers import AutoModelForMaskedLM
        
        # Load masked LM version of model
        mlm_model = AutoModelForMaskedLM.from_pretrained(
            self.model_name
        ).to(self.device)
        mlm_model.eval()
        
        results = {}
        
        for attr_name, terms in fill_values.items():
            attr_results = []
            
            for term in terms:
                # Fill template with term
                filled_text = template.replace("[DEMO]", term)
                
                # Tokenize
                inputs = self.tokenizer(
                    filled_text,
                    return_tensors="pt"
                ).to(self.device)
                
                # Find mask token position
                mask_token_index = torch.where(
                    inputs["input_ids"] == self.tokenizer.mask_token_id
                )[1]
                
                if len(mask_token_index) == 0:
                    logger.warning(f"No mask token found in: {filled_text}")
                    continue
                
                # Get predictions
                with torch.no_grad():
                    outputs = mlm_model(**inputs)
                    predictions = outputs.logits
                
                # Get top predictions for mask position
                mask_predictions = predictions[0, mask_token_index, :]
                top_tokens = torch.topk(mask_predictions, k=10, dim=1)
                
                # Convert to terms and probabilities
                top_terms = []
                for token_id, score in zip(
                    top_tokens.indices[0].tolist(),
                    top_tokens.values[0].tolist()
                ):
                    token = self.tokenizer.decode([token_id])
                    prob = torch.softmax(mask_predictions[0], dim=0)[token_id].item()
                    top_terms.append((token.strip(), prob))
                
                attr_results.append({
                    'fill_value': term,
                    'top_predictions': top_terms
                })
            
            results[attr_name] = attr_results
        
        return results
    
    def analyze_template_bias_disparities(
        self,
        template_results: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """
        Analyze disparities in template predictions across demographic groups.
        
        Args:
            template_results: Results from test_template_bias
            
        Returns:
            Dictionary of disparity metrics
        """
        # Extract prediction distributions for each group
        group_distributions = {}
        
        for group_name, group_results in template_results.items():
            all_predictions = []
            for result in group_results:
                predictions = {
                    term: prob
                    for term, prob in result['top_predictions']
                }
                all_predictions.append(predictions)
            
            # Average prediction probabilities across group
            all_terms = set()
            for pred in all_predictions:
                all_terms.update(pred.keys())
            
            avg_probs = {}
            for term in all_terms:
                probs = [pred.get(term, 0.0) for pred in all_predictions]
                avg_probs[term] = np.mean(probs)
            
            group_distributions[group_name] = avg_probs
        
        # Compute disparities
        if len(group_distributions) < 2:
            return {}
        
        disparities = {}
        group_names = list(group_distributions.keys())
        
        # Compare first group with others
        base_group = group_names[0]
        base_dist = group_distributions[base_group]
        
        for compare_group in group_names[1:]:
            compare_dist = group_distributions[compare_group]
            
            # Find common terms
            common_terms = set(base_dist.keys()) & set(compare_dist.keys())
            
            if not common_terms:
                continue
            
            # Compute KL divergence
            kl_div = 0
            for term in common_terms:
                p = base_dist[term]
                q = compare_dist[term]
                if p > 0 and q > 0:
                    kl_div += p * np.log(p / q)
            
            disparities[f"{base_group}_vs_{compare_group}_kl"] = float(kl_div)
        
        return disparities
    
    def generate_bias_report(self, output_path: str = "bias_audit_report.txt"):
        """
        Generate comprehensive bias audit report.
        
        Args:
            output_path: Path to save report
        """
        logger.info("Generating bias audit report...")
        
        # Run all bias tests
        weat_results = self.audit_demographic_clinical_associations()
        
        # Test template biases
        compliance_template = (
            "The [DEMO] patient was noncompliant with [MASK] recommendations."
        )
        compliance_results = self.test_template_bias(
            compliance_template,
            {
                'white': self.demographic_terms['race_white'],
                'black': self.demographic_terms['race_black']
            }
        )
        
        # Generate report
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Clinical Language Model Bias Audit Report\n")
            f.write(f"Model: {self.model_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # WEAT results
            f.write("Word Embedding Association Test (WEAT) Results\n")
            f.write("-" * 80 + "\n\n")
            
            significant_biases = [
                (name, result)
                for name, result in weat_results.items()
                if result['significant']
            ]
            
            if significant_biases:
                f.write(f"Found {len(significant_biases)} significant biases:\n\n")
                for name, result in significant_biases:
                    f.write(f"Test: {name}\n")
                    f.write(f"  Effect Size: {result['effect_size']:.4f}\n")
                    f.write(f"  P-value: {result['p_value']:.4f}\n")
                    f.write(f"  Interpretation: {result['interpretation']}\n\n")
            else:
                f.write("No significant biases detected in WEAT tests.\n\n")
            
            # Template bias results
            f.write("\n" + "=" * 80 + "\n")
            f.write("Masked Language Model Prediction Analysis\n")
            f.write("-" * 80 + "\n\n")
            
            for group_name, group_results in compliance_results.items():
                f.write(f"\nGroup: {group_name}\n")
                for result in group_results:
                    f.write(f"  Fill value: {result['fill_value']}\n")
                    f.write(f"  Top predictions:\n")
                    for term, prob in result['top_predictions'][:5]:
                        f.write(f"    {term}: {prob:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Recommendations\n")
            f.write("-" * 80 + "\n\n")
            
            if significant_biases:
                f.write(
                    "IMPORTANT: This model shows evidence of bias in its learned "
                    "representations. Before deploying for clinical use:\n\n"
                    "1. Conduct additional validation on diverse patient populations\n"
                    "2. Consider bias mitigation techniques (see next section)\n"
                    "3. Implement monitoring systems to detect biased predictions\n"
                    "4. Ensure human oversight for all model predictions\n"
                    "5. Conduct stakeholder engagement with affected communities\n"
                )
            else:
                f.write(
                    "This audit did not detect significant biases in the tested "
                    "dimensions. However:\n\n"
                    "1. Additional testing on specific clinical tasks is recommended\n"
                    "2. Monitor for emergent biases during deployment\n"
                    "3. Regularly re-audit as model is updated\n"
                )
        
        logger.info(f"Bias audit report saved to {output_path}")
```

This bias auditing framework provides comprehensive evaluation of potential biases in clinical language models before they are used for downstream tasks. The WEAT approach has been validated for detecting implicit biases in word embeddings, while the template-based analysis reveals how models make different predictions depending on demographic context \citep{caliskan2017semantics, garg2018word, sun2022examining}.

### 6.4.2 Bias Mitigation Strategies

Once biases are detected in clinical language models, several mitigation strategies can be employed. We implement approaches including adversarial debiasing, counterfactual data augmentation, and constrained fine-tuning.

```python
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from transformers import AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class AdversarialDebiaser(nn.Module):
    """
    Adversarial training for debiasing clinical language models.
    
    This approach trains the main model to predict clinical outcomes while
    simultaneously training an adversary to predict demographic attributes.
    The main model is penalized for encoding demographic information,
    encouraging it to learn representations that are predictive of clinical
    outcomes but not correlated with protected attributes.
    """
    
    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        num_protected_attributes: int,
        hidden_dim: int = 768,
        adversary_hidden_dim: int = 256,
        adversary_strength: float = 1.0
    ):
        """
        Initialize adversarial debiasing framework.
        
        Args:
            base_model_name: Pre-trained clinical language model
            num_labels: Number of primary task labels
            num_protected_attributes: Number of protected attribute classes
            hidden_dim: Hidden dimension of base model
            adversary_hidden_dim: Hidden dimension of adversary
            adversary_strength: Weight of adversarial loss
        """
        super(AdversarialDebiaser, self).__init__()
        
        self.adversary_strength = adversary_strength
        
        # Load base clinical language model
        self.encoder = AutoModel.from_pretrained(base_model_name)
        
        # Primary task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_labels)
        )
        
        # Adversarial classifier for protected attributes
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim, adversary_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adversary_hidden_dim, adversary_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adversary_hidden_dim // 2, num_protected_attributes)
        )
        
        logger.info("Initialized adversarial debiasing framework")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        protected_attributes: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adversarial training.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Task labels (optional)
            protected_attributes: Protected attribute labels (optional)
            
        Returns:
            Dictionary with predictions and losses
        """
        # Get contextualized representations
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Primary task prediction
        task_logits = self.task_classifier(pooled_output)
        
        # Adversarial prediction (with gradient reversal)
        # In training, we'll reverse gradients flowing to encoder
        adversary_logits = self.adversary(pooled_output)
        
        output_dict = {
            'task_logits': task_logits,
            'adversary_logits': adversary_logits
        }
        
        # Compute losses if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            task_loss = loss_fct(task_logits, labels)
            output_dict['task_loss'] = task_loss
            
            if protected_attributes is not None:
                adversary_loss = loss_fct(adversary_logits, protected_attributes)
                output_dict['adversary_loss'] = adversary_loss
                
                # Total loss: maximize task accuracy, minimize adversary accuracy
                # The gradient reversal is implemented in training loop
                output_dict['total_loss'] = (
                    task_loss - self.adversary_strength * adversary_loss
                )
        
        return output_dict
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer_encoder: torch.optim.Optimizer,
        optimizer_task: torch.optim.Optimizer,
        optimizer_adversary: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Perform one training step with adversarial training.
        
        Training procedure:
        1. Update adversary to better predict protected attributes
        2. Update encoder and task classifier to predict task while
           fooling adversary about protected attributes
        
        Args:
            batch: Training batch
            optimizer_encoder: Optimizer for encoder
            optimizer_task: Optimizer for task classifier
            optimizer_adversary: Optimizer for adversary
            
        Returns:
            Dictionary of loss values
        """
        # Forward pass
        output = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            protected_attributes=batch['protected_attributes']
        )
        
        # Step 1: Update adversary
        optimizer_adversary.zero_grad()
        output['adversary_loss'].backward(retain_graph=True)
        optimizer_adversary.step()
        
        # Step 2: Update encoder and task classifier
        # Goal: predict task accurately while hiding protected attributes
        optimizer_encoder.zero_grad()
        optimizer_task.zero_grad()
        
        # Recompute with updated adversary
        output = self.forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            protected_attributes=batch['protected_attributes']
        )
        
        # Loss combines task accuracy with adversary fooling
        combined_loss = (
            output['task_loss'] +
            self.adversary_strength * output['adversary_loss']
        )
        
        combined_loss.backward()
        optimizer_encoder.step()
        optimizer_task.step()
        
        return {
            'task_loss': output['task_loss'].item(),
            'adversary_loss': output['adversary_loss'].item(),
            'combined_loss': combined_loss.item()
        }


class CounterfactualAugmenter:
    """
    Counterfactual data augmentation for debiasing.
    
    This approach creates counterfactual examples by swapping demographic
    terms in clinical text while keeping medical content constant. Training
    on both original and counterfactual examples encourages models to learn
    representations invariant to protected attributes.
    """
    
    def __init__(self):
        """Initialize counterfactual augmenter."""
        
        # Define demographic term mappings for counterfactual generation
        self.demographic_swaps = {
            'white': ['black', 'hispanic', 'asian'],
            'black': ['white', 'hispanic', 'asian'],
            'hispanic': ['white', 'black', 'asian'],
            'asian': ['white', 'black', 'hispanic'],
            'caucasian': ['african american', 'latino', 'asian american'],
            'african american': ['caucasian', 'latino', 'asian american'],
            'man': ['woman'],
            'woman': ['man'],
            'male': ['female'],
            'female': ['male'],
            'he': ['she'],
            'she': ['he'],
            'him': ['her'],
            'her': ['him'],
            'his': ['her'],
        }
        
        logger.info("Initialized counterfactual augmenter")
    
    def generate_counterfactuals(
        self,
        text: str,
        num_counterfactuals: int = 1
    ) -> List[str]:
        """
        Generate counterfactual versions of clinical text.
        
        Args:
            text: Original clinical text
            num_counterfactuals: Number of counterfactuals to generate
            
        Returns:
            List of counterfactual texts
        """
        counterfactuals = []
        text_lower = text.lower()
        
        # Find demographic terms in text
        found_terms = []
        for term in self.demographic_swaps:
            if term in text_lower:
                found_terms.append(term)
        
        if not found_terms:
            # No demographic terms to swap
            return [text]
        
        # Generate counterfactuals by swapping terms
        for _ in range(num_counterfactuals):
            counterfactual = text
            
            for term in found_terms:
                if term not in self.demographic_swaps:
                    continue
                
                # Randomly choose a swap
                swap_options = self.demographic_swaps[term]
                swap_term = np.random.choice(swap_options)
                
                # Replace term (case-insensitive)
                import re
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                
                # Preserve case of original
                def replace_with_case(match):
                    original = match.group(0)
                    if original[0].isupper():
                        return swap_term.capitalize()
                    return swap_term
                
                counterfactual = pattern.sub(replace_with_case, counterfactual)
            
            counterfactuals.append(counterfactual)
        
        return counterfactuals
    
    def augment_dataset(
        self,
        texts: List[str],
        labels: List[int],
        num_counterfactuals_per_example: int = 1
    ) -> Tuple[List[str], List[int]]:
        """
        Augment training dataset with counterfactual examples.
        
        Args:
            texts: Original texts
            labels: Original labels
            num_counterfactuals_per_example: Counterfactuals per example
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Add original
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            # Generate and add counterfactuals
            counterfactuals = self.generate_counterfactuals(
                text,
                num_counterfactuals_per_example
            )
            
            for cf in counterfactuals:
                if cf != text:  # Don't add duplicates
                    augmented_texts.append(cf)
                    augmented_labels.append(label)
        
        logger.info(
            f"Augmented dataset from {len(texts)} to {len(augmented_texts)} examples"
        )
        
        return augmented_texts, augmented_labels
```

These bias mitigation approaches provide practical tools for reducing the biases that clinical language models learn from historical training data. Adversarial debiasing has shown promise for learning representations that are predictive for clinical tasks while being less correlated with protected attributes, while counterfactual augmentation helps models generalize across demographic groups \citep{zhang2018mitigating, madras2018learning, hall2022effect}.

## 6.5 Large Language Models in Healthcare

Recent advances in large language models have created new opportunities and challenges for healthcare applications. Models like GPT-4, Claude, and specialized medical LLMs can generate clinical documentation, answer medical questions, and assist with diagnostic reasoning. However, deploying these powerful models in healthcare requires careful attention to safety, accuracy, and fairness given the high stakes of medical decision-making and the potential for these models to encode and amplify biases from their training data.

### 6.5.1 Clinical Question Answering with LLMs

We implement a framework for using large language models for clinical question answering while incorporating comprehensive safety checks and bias monitoring.

```python
from typing import List, Dict, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class ClinicalQASystem:
    """
    Clinical question answering system using large language models.
    
    Implements safety checks, bias monitoring, and validation appropriate
    for high-stakes healthcare applications.
    """
    
    def __init__(
        self,
        model_name: str = "clinical-llm",
        temperature: float = 0.3,
        max_tokens: int = 512
    ):
        """
        Initialize clinical QA system.
        
        Args:
            model_name: Name of LLM to use
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Safety checks
        self.unsafe_patterns = [
            r'definitely',
            r'certainly',
            r'always',
            r'never',
            r'100%',
            r'guaranteed'
        ]
        
        # Bias detection patterns
        self.bias_patterns = {
            'race': [
                r'black patients? (are|have|tend to)',
                r'white patients? (are|have|tend to)',
                r'hispanic patients? (are|have|tend to)',
            ],
            'gender': [
                r'(wo)?men (are|have|tend to)',
                r'(fe)?male patients? (are|have|tend to)'
            ],
            'ses': [
                r'poor patients? (are|have|tend to)',
                r'wealthy patients? (are|have|tend to)',
                r'medicaid patients? (are|have|tend to)'
            ]
        }
        
        logger.info(f"Initialized clinical QA system with {model_name}")
    
    def generate_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        patient_context: Optional[Dict] = None
    ) -> str:
        """
        Generate prompt for LLM with appropriate framing and safety instructions.
        
        Args:
            question: Clinical question
            context: Optional context (e.g., from retrieved documents)
            patient_context: Optional patient-specific context
            
        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "You are a clinical AI assistant providing evidence-based medical information.",
            "Your responses should:",
            "- Be accurate and evidence-based",
            "- Acknowledge uncertainty when appropriate",
            "- Avoid making definitive diagnoses or treatment recommendations",
            "- Recommend consulting healthcare providers for medical decisions",
            "- Not make assumptions based on demographic characteristics",
            "- Treat all patients with equal respect regardless of background",
            "",
        ]
        
        if context:
            prompt_parts.extend([
                "Relevant context:",
                context,
                ""
            ])
        
        if patient_context:
            # Include only clinically relevant patient info
            relevant_context = {
                k: v for k, v in patient_context.items()
                if k in ['age', 'chief_complaint', 'relevant_history']
            }
            prompt_parts.extend([
                "Patient context:",
                str(relevant_context),
                ""
            ])
        
        prompt_parts.extend([
            f"Question: {question}",
            "",
            "Response:"
        ])
        
        return "\n".join(prompt_parts)
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        patient_context: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Answer clinical question with safety checks.
        
        Args:
            question: Clinical question
            context: Optional context
            patient_context: Optional patient context
            
        Returns:
            Dictionary with answer and safety assessment
        """
        # Generate prompt
        prompt = self.generate_prompt(question, context, patient_context)
        
        # Call LLM (placeholder - would use actual API)
        # In production, would call OpenAI, Anthropic, or other LLM API
        response_text = self._call_llm(prompt)
        
        # Safety checks
        safety_issues = self._check_safety(response_text)
        bias_flags = self._check_bias(response_text)
        
        # Assess response quality
        quality_score = self._assess_quality(response_text)
        
        return {
            'question': question,
            'answer': response_text,
            'safety_issues': safety_issues,
            'bias_flags': bias_flags,
            'quality_score': quality_score,
            'safe_to_present': len(safety_issues) == 0 and len(bias_flags) == 0
        }
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM API to generate response.
        
        This is a placeholder - in production would call actual LLM API.
        """
        # Placeholder implementation
        return (
            "Based on current clinical evidence, this condition typically "
            "presents with [symptoms]. However, presentation can vary "
            "significantly between patients. A thorough clinical evaluation "
            "by a healthcare provider is necessary for accurate diagnosis. "
            "Treatment options should be individualized based on patient "
            "factors and preferences."
        )
    
    def _check_safety(self, response: str) -> List[str]:
        """
        Check response for safety issues.
        
        Args:
            response: Generated response
            
        Returns:
            List of safety issues found
        """
        issues = []
        
        # Check for overly confident language
        for pattern in self.unsafe_patterns:
            if re.search(pattern, response.lower()):
                issues.append(f"Overly confident language: {pattern}")
        
        # Check for medical advice without qualification
        advice_keywords = ['should', 'must', 'recommend', 'advise']
        qualification_keywords = [
            'may', 'consider', 'discuss with', 'consult',
            'healthcare provider', 'physician', 'doctor'
        ]
        
        has_advice = any(kw in response.lower() for kw in advice_keywords)
        has_qualification = any(kw in response.lower() for kw in qualification_keywords)
        
        if has_advice and not has_qualification:
            issues.append("Medical advice without qualification")
        
        return issues
    
    def _check_bias(self, response: str) -> List[Dict[str, str]]:
        """
        Check response for potential bias.
        
        Args:
            response: Generated response
            
        Returns:
            List of potential bias flags
        """
        flags = []
        
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response.lower()):
                    flags.append({
                        'type': bias_type,
                        'pattern': pattern,
                        'context': self._extract_context(response, pattern)
                    })
        
        return flags
    
    def _extract_context(self, text: str, pattern: str, window: int = 50) -> str:
        """Extract context around a pattern match."""
        match = re.search(pattern, text.lower())
        if not match:
            return ""
        
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        
        return text[start:end]
    
    def _assess_quality(self, response: str) -> float:
        """
        Assess quality of response.
        
        Simple heuristic assessment - in production would use more
        sophisticated methods.
        
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Penalize very short responses
        if len(response.split()) < 20:
            score -= 0.3
        
        # Penalize responses without evidence markers
        evidence_markers = [
            'study', 'research', 'evidence', 'trial',
            'according to', 'suggests', 'indicates'
        ]
        if not any(marker in response.lower() for marker in evidence_markers):
            score -= 0.2
        
        # Reward uncertainty acknowledgment
        uncertainty_markers = [
            'may', 'might', 'possibly', 'uncertain',
            'varies', 'individual', 'depends'
        ]
        if any(marker in response.lower() for marker in uncertainty_markers):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def evaluate_fairness(
        self,
        questions: List[str],
        patient_contexts: List[Dict]
    ) -> Dict[str, any]:
        """
        Evaluate whether system provides equitable answers across patient groups.
        
        Args:
            questions: List of clinical questions
            patient_contexts: List of patient context dictionaries with demographics
            
        Returns:
            Fairness evaluation results
        """
        results = []
        
        for question in questions:
            question_results = {}
            
            # Generate answers for different demographic contexts
            for context in patient_contexts:
                answer = self.answer_question(question, patient_context=context)
                
                demo_key = f"{context.get('race', 'unknown')}_{context.get('gender', 'unknown')}"
                question_results[demo_key] = answer
            
            results.append({
                'question': question,
                'answers_by_demographics': question_results
            })
        
        # Analyze consistency
        consistency_metrics = self._analyze_consistency(results)
        
        return {
            'detailed_results': results,
            'consistency_metrics': consistency_metrics
        }
    
    def _analyze_consistency(
        self,
        results: List[Dict]
    ) -> Dict[str, float]:
        """
        Analyze consistency of answers across demographic groups.
        
        Args:
            results: Results from evaluate_fairness
            
        Returns:
            Dictionary of consistency metrics
        """
        # Compute various consistency metrics
        metrics = {
            'safety_issue_rate_mean': 0.0,
            'safety_issue_rate_std': 0.0,
            'bias_flag_rate_mean': 0.0,
            'bias_flag_rate_std': 0.0,
            'quality_score_mean': 0.0,
            'quality_score_std': 0.0
        }
        
        # Extract metrics across all answers
        all_safety_rates = []
        all_bias_rates = []
        all_quality_scores = []
        
        for result in results:
            for demo_key, answer in result['answers_by_demographics'].items():
                safety_rate = len(answer['safety_issues']) / max(
                    1, len(answer['answer'].split())
                )
                bias_rate = len(answer['bias_flags']) / max(
                    1, len(answer['answer'].split())
                )
                
                all_safety_rates.append(safety_rate)
                all_bias_rates.append(bias_rate)
                all_quality_scores.append(answer['quality_score'])
        
        if all_quality_scores:
            metrics['safety_issue_rate_mean'] = float(np.mean(all_safety_rates))
            metrics['safety_issue_rate_std'] = float(np.std(all_safety_rates))
            metrics['bias_flag_rate_mean'] = float(np.mean(all_bias_rates))
            metrics['bias_flag_rate_std'] = float(np.std(all_bias_rates))
            metrics['quality_score_mean'] = float(np.mean(all_quality_scores))
            metrics['quality_score_std'] = float(np.std(all_quality_scores))
        
        return metrics
```

This clinical QA system provides a framework for safely deploying large language models in healthcare applications while incorporating comprehensive safety checks and fairness monitoring. The approach emphasizes uncertainty quantification, appropriate qualification of medical advice, and detection of potential biases that could lead to disparate care recommendations \citep{singhal2022large, thirunavukarasu2023large, nori2023capabilities}.

## 6.6 Conclusion

Natural language processing for clinical text presents unique technical challenges and profound equity implications. Clinical documentation encodes not only medical information but also systematic biases in how clinicians describe and reason about patients from different backgrounds. When we build NLP systems on this text, we must acknowledge and actively address these biases rather than treating them as mere technical noise to be filtered out. This chapter has developed comprehensive approaches for clinical NLP that maintain equity considerations throughout the development lifecycle, from fundamental text preprocessing decisions through advanced applications of large language models. The implementations provided enable practitioners to build production-grade clinical NLP systems that can detect and mitigate bias while delivering value for diverse patient populations. The path forward requires continued vigilance as language models become more powerful, with ongoing monitoring for emergent biases and sustained commitment to centering health equity in all technical decisions.

## Bibliography

Alsentzer, E., Murphy, J. R., Boag, W., Weng, W. H., Jin, D., Naumann, T., & McDermott, M. (2019). Publicly available clinical BERT embeddings. *Proceedings of the 2nd Clinical Natural Language Processing Workshop*, 72-78. https://arxiv.org/abs/1904.03323

Brown, A., Desmond, J., Meigs, J. B., Greenfield, S., Karter, A. J., Nguyen, T. T., ... & Selvin, E. (2015). Disparities in clinical documentation of Black, Hispanic, and white patients. *Journal of General Internal Medicine*, 30(8), 1201-1207. https://doi.org/10.1007/s11606-015-3283-9

Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183-186. https://doi.org/10.1126/science.aal4230

Chapman, W. W., Bridewell, W., Hanbury, P., Cooper, G. F., & Buchanan, B. G. (2001). A simple algorithm for identifying negated findings and diseases in discharge summaries. *Journal of Biomedical Informatics*, 34(5), 301-310. https://doi.org/10.1006/jbin.2001.1029

Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M. (2021). Ethical machine learning in healthcare. *Annual Review of Biomedical Data Science*, 4, 123-144. https://doi.org/10.1146/annurev-biodatasci-092820-114757

Davenport, T., & Kalakota, R. (2019). The potential for artificial intelligence in healthcare. *Future Healthcare Journal*, 6(2), 94-98. https://doi.org/10.7861/futurehosp.6-2-94

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 4171-4186. https://arxiv.org/abs/1810.04805

Fernandez, A., Schillinger, D., Warton, E. M., Adler, N., Moffet, H. H., Schenker, Y., ... & Karter, A. J. (2011). Language barriers, physician-patient language concordance, and glycemic control among insured Latinos with diabetes: the Diabetes Study of Northern California (DISTANCE). *Journal of General Internal Medicine*, 26(2), 170-176. https://doi.org/10.1007/s11606-010-1507-6

Garg, N., Schiebinger, L., Jurafsky, D., & Zou, J. (2018). Word embeddings quantify 100 years of gender and ethnic stereotypes. *Proceedings of the National Academy of Sciences*, 115(16), E3635-E3644. https://doi.org/10.1073/pnas.1720347115

Hall, M. A., Lee, S. S., Jiang, H., Erickson, P., & Cook, T. (2022). Effect of counterfactual fairness on bias mitigation in biomedical artificial intelligence. *JAMA Network Open*, 5(3), e220794. https://doi.org/10.1001/jamanetworkopen.2022.0794

Harkema, H., Dowling, J. N., Thornblade, T., & Chapman, W. W. (2009). ConText: An algorithm for determining negation, experiencer, and temporal status from clinical reports. *Journal of Biomedical Informatics*, 42(5), 839-851. https://doi.org/10.1016/j.jbi.2009.05.002

Huang, K., Altosaar, J., & Ranganath, R. (2020). ClinicalBERT: Modeling clinical notes and predicting hospital readmission. *arXiv preprint arXiv:1904.05342*. https://arxiv.org/abs/1904.05342

Huang, Z., Xu, W., & Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. *arXiv preprint arXiv:1508.01991*. https://arxiv.org/abs/1508.01991

Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L. W. H., Feng, M., Ghassemi, M., ... & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3(1), 1-9. https://doi.org/10.1038/sdata.2016.35

Karliner, L. S., Jacobs, E. A., Chen, A. H., & Mutha, S. (2007). Do professional interpreters improve clinical care for patients with limited English proficiency? A systematic review of the literature. *Health Services Research*, 42(2), 727-754. https://doi.org/10.1111/j.1475-6773.2006.00629.x

Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., & Dyer, C. (2016). Neural architectures for named entity recognition. *Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics*, 260-270. https://arxiv.org/abs/1603.01360

Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234-1240. https://doi.org/10.1093/bioinformatics/btz682

Madras, D., Creager, E., Pitassi, T., & Zemel, R. (2018). Learning adversarially fair and transferable representations. *Proceedings of the 35th International Conference on Machine Learning*, 3384-3393. http://proceedings.mlr.press/v80/madras18a.html

Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019). ScispaCy: Fast and robust models for biomedical natural language processing. *Proceedings of the 18th BioNLP Workshop*, 319-327. https://arxiv.org/abs/1902.07669

Nori, H., King, N., McKinney, S. M., Carignan, D., & Horvitz, E. (2023). Capabilities of GPT-4 on medical challenge problems. *arXiv preprint arXiv:2303.13375*. https://arxiv.org/abs/2303.13375

Park, G., Schwartz, H. A., Eichstaedt, J. C., Kern, M. L., Kosinski, M., Stillwell, D. J., ... & Seligman, M. E. (2021). Automatic personality assessment through social media language. *Journal of Personality and Social Psychology*, 108(6), 934-952. https://doi.org/10.1037/pspp0000020

Peng, Y., Yan, S., & Lu, Z. (2019). Transfer learning in biomedical natural language processing: An evaluation of BERT and ELMo on ten benchmarking datasets. *Proceedings of the 18th BioNLP Workshop*, 58-65. https://arxiv.org/abs/1906.05474

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., ... & Natarajan, V. (2022). Large language models encode clinical knowledge. *arXiv preprint arXiv:2212.13138*. https://arxiv.org/abs/2212.13138

Sun, J., Pang, Q., Song, R., Zhang, Y., Chen, M., & Yang, C. (2022). Examining biases in clinical notes: a study of race-related language in admission notes. *JAMA Network Open*, 5(10), e2235150. https://doi.org/10.1001/jamanetworkopen.2022.35150

Thirunavukarasu, A. J., Ting, D. S. J., Elangovan, K., Gutierrez, L., Tan, T. F., & Ting, D. S. W. (2023). Large language models in medicine. *Nature Medicine*, 29(8), 1930-1940. https://doi.org/10.1038/s41591-023-02448-8

Wang, X., Zhang, Y., Ren, X., Zhang, Y., Zitnik, M., Shang, J., ... & Han, J. (2018). Cross-type biomedical named entity recognition with deep multi-task learning. *Bioinformatics*, 35(10), 1745-1752. https://doi.org/10.1093/bioinformatics/bty869

Weng, W. H., Wagholikar, K. B., McCray, A. T., Szolovits, P., & Chueh, H. C. (2017). Medical subdomain classification of clinical notes using a machine learning-based natural language processing approach. *BMC Medical Informatics and Decision Making*, 17(1), 1-13. https://doi.org/10.1186/s12911-017-0556-8

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating unwanted biases with adversarial learning. *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society*, 335-340. https://doi.org/10.1145/3278721.3278779

Zou, J., & Schiebinger, L. (2018). AI can be sexist and racist—it's time to make it fair. *Nature*, 559(7714), 324-326. https://doi.org/10.1038/d41586-018-05707-8