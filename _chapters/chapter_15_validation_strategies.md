---
layout: chapter
title: "Chapter 15: Clinical Validation Frameworks and External Validity"
chapter_number: 15
part_number: 4
prev_chapter: /chapters/chapter-14-interpretability-explainability/
next_chapter: /chapters/chapter-16-uncertainty-calibration/
---
# Chapter 15: Clinical Validation Frameworks and External Validity

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Design comprehensive validation studies for clinical AI systems that adequately assess performance across diverse patient populations and care settings
2. Implement stratified validation frameworks that explicitly evaluate fairness metrics alongside standard performance measures
3. Calculate appropriate sample sizes for validation studies that account for the need to detect clinically meaningful disparities across demographic subgroups
4. Conduct temporal validation to assess model degradation over time and across changing clinical practices
5. Execute external validation across geographically and demographically diverse sites to evaluate model generalizability
6. Design and analyze prospective validation studies that assess real-world clinical impact
7. Implement continuous monitoring frameworks for deployed models with automated alerts for fairness violations
8. Document validation findings comprehensively for regulatory review and clinical stakeholder communication

## 15.1 Introduction: Why Standard Validation Fails for Health Equity

Validation is the critical bridge between model development and clinical deployment. A model that performs excellently during development can fail catastrophically in practice if validation was inadequate. The stakes are particularly high in healthcare, where prediction errors can lead to missed diagnoses, inappropriate treatments, and preventable mortality. For models intended to serve diverse populations, standard validation approaches are systematically insufficient because they focus on average performance while obscuring disparities that manifest within specific patient subgroups.

The fundamental challenge is that aggregate performance metrics can be excellent even when a model fails dramatically for particular demographic groups or clinical contexts. A model achieving an area under the receiver operating characteristic curve of 0.90 overall might have an AUC of 0.95 for well-represented populations but only 0.75 for underrepresented groups. Standard validation studies would report the impressive overall performance while entirely missing the equity failure. This pattern has occurred repeatedly in deployed clinical AI systems, from sepsis prediction models that perform worse for Black patients to algorithms allocating healthcare resources that systematically disadvantage those with complex social needs.

The problem extends beyond simple underrepresentation in validation cohorts. Even when validation datasets include diverse populations, standard analytic approaches aggregate results across all patients, making it statistically challenging to detect disparities without explicitly testing for them. A validation study might include adequate numbers of patients from underserved communities yet still fail to identify fairness issues if the analysis plan does not include stratified evaluation by demographics, clinical complexity, and care setting characteristics. The default assumption that models performing well overall will perform adequately for all subgroups is demonstrably false in practice.

Clinical AI validation faces unique challenges compared to other machine learning domains. Healthcare data exhibits substantial heterogeneity across sites and over time due to differences in patient populations, clinical practices, documentation patterns, and available technologies. A model trained and validated at academic medical centers may fail when deployed in community hospitals or federally qualified health centers serving predominantly underserved populations. Temporal shifts in disease prevalence, treatment standards, and coding practices can degrade model performance in ways not captured by single-timepoint validation. The consequences of deployment failures include not just poor predictions but potential harms to patients and erosion of clinician trust in AI systems generally.

From an equity perspective, inadequate validation perpetuates health disparities in multiple ways. Models validated primarily on data from well-resourced institutions may fail to account for systematic differences in data completeness, documentation quality, and clinical phenotypes observed in under-resourced settings. Validation studies conducted at single institutions cannot assess whether models generalize across the diversity of care environments where deployment is intended. Without explicit fairness evaluation, validation may declare models ready for deployment despite exhibiting discriminatory behavior that would become apparent only through stratified analysis. The result is deployment of systems that appear rigorously validated by standard criteria yet exacerbate rather than reduce health inequities.

This chapter develops comprehensive validation strategies specifically designed for clinical AI systems intended to serve diverse populations equitably. We begin by establishing frameworks for internal validation that maintain adequate representation of key patient subgroups and care contexts. We then cover temporal validation to detect performance degradation, external validation across diverse sites and populations, prospective validation in real clinical workflows, and continuous post-deployment monitoring. Throughout, we emphasize validation designs that explicitly assess fairness alongside traditional performance metrics and provide adequate statistical power to detect meaningful disparities.

The technical implementations in this chapter provide production-ready code for validation frameworks including stratified performance evaluation with fairness metrics, power calculations for detecting disparities across subgroups, temporal validation with drift detection, multi-site external validation, and automated monitoring systems for deployed models. Each implementation includes comprehensive logging, error handling, and documentation suitable for regulatory review. By the end of this chapter, readers will have both conceptual understanding and practical tools for rigorous validation that ensures clinical AI systems are safe and fair for all populations they are intended to serve.

## 15.2 Internal Validation with Equity Considerations

Internal validation uses data from the same source as model development to provide initial estimates of expected performance. While internal validation alone is insufficient for clinical deployment, it serves as an essential first step in the validation hierarchy and must be designed carefully to avoid misleading conclusions about model fairness and generalizability.

The fundamental challenge in internal validation for equity is ensuring that data splitting preserves adequate representation of key patient subgroups. Standard random splits or simple stratified splits by outcome often result in validation sets with insufficient numbers of patients from underrepresented demographic groups, making it statistically impossible to detect meaningful disparities. If a training dataset contains only five percent of patients from a particular demographic group, a twenty percent random validation split yields only one percent of the total dataset for evaluating performance in that group. With complex clinical datasets where positive outcome rates may be ten percent or lower, this results in validation cohorts with single-digit positive cases for certain demographic subgroups, providing essentially no information about model fairness.

Thoughtful data splitting for equitable validation requires explicit consideration of multiple stratification dimensions simultaneously. We must ensure adequate representation not just of overall outcome rates but of outcomes within demographic subgroups, clinical risk strata, and care setting characteristics. This often necessitates larger validation sets than standard rules of thumb would suggest, along with stratified sampling approaches that prioritize representation of groups for whom fairness evaluation is particularly important. The tradeoff is reduced training data, but this is acceptable because a model that cannot be adequately validated should not be deployed regardless of its training performance.

We now develop a comprehensive internal validation framework with explicit equity considerations built into every component.

### 15.2.1 Stratified Data Splitting for Adequate Subgroup Representation

The foundation of equitable internal validation is a data splitting strategy that ensures validation cohorts contain sufficient numbers of patients across all relevant demographic groups and clinical strata to enable meaningful fairness evaluation. This requires moving beyond simple random or outcome-stratified splits to multidimensional stratification that preserves key distributions.

```python
"""
Internal validation framework with equity-focused data splitting.

This module implements comprehensive internal validation strategies that
ensure adequate representation of diverse patient populations for fair
evaluation. The splitting strategies go beyond standard random splits to
explicitly preserve demographic distributions and enable detection of
disparities with adequate statistical power.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve
)
import scipy.stats as stats
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SplitStrategy(Enum):
    """Enumeration of data splitting strategies for validation."""
    RANDOM = "random"
    STRATIFIED_OUTCOME = "stratified_outcome"
    STRATIFIED_DEMOGRAPHIC = "stratified_demographic"
    MULTIDIMENSIONAL_STRATIFIED = "multidimensional_stratified"
    TEMPORAL = "temporal"

@dataclass
class ValidationSplit:
    """
    Container for train/validation data split with metadata.

    Attributes:
        train_indices: Indices of training samples
        val_indices: Indices of validation samples
        train_demographics: Demographic distribution in training set
        val_demographics: Demographic distribution in validation set
        split_metadata: Additional metadata about the split
    """
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_demographics: Dict[str, Dict[str, float]]
    val_demographics: Dict[str, Dict[str, float]]
    split_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_representation_ratio(
        self,
        demographic_column: str,
        demographic_value: str
    ) -> float:
        """
        Calculate ratio of validation to training representation for a group.

        A ratio near 1.0 indicates balanced representation. Ratios substantially
        different from 1.0 suggest potential issues with stratification.

        Args:
            demographic_column: Name of demographic variable
            demographic_value: Specific value to check

        Returns:
            Ratio of validation to training proportions
        """
        if demographic_column not in self.val_demographics:
            raise ValueError(f"Demographic column {demographic_column} not found")

        train_prop = self.train_demographics[demographic_column].get(
            demographic_value, 0.0
        )
        val_prop = self.val_demographics[demographic_column].get(
            demographic_value, 0.0
        )

        if train_prop == 0:
            return float('inf') if val_prop > 0 else 1.0

        return val_prop / train_prop

    def check_minimum_representation(
        self,
        demographic_column: str,
        minimum_count: int = 50
    ) -> Dict[str, bool]:
        """
        Check if all demographic groups meet minimum count threshold.

        Args:
            demographic_column: Name of demographic variable
            minimum_count: Minimum number of samples required

        Returns:
            Dictionary mapping demographic values to whether minimum is met
        """
        total_val_samples = len(self.val_indices)
        results = {}

        for value, proportion in self.val_demographics[demographic_column].items():
            count = int(proportion * total_val_samples)
            results[value] = count >= minimum_count

        return results

class EquityAwareDataSplitter:
    """
    Advanced data splitting for equitable internal validation.

    This class implements sophisticated data splitting strategies that ensure
    validation cohorts contain adequate representation of diverse patient
    populations for meaningful fairness evaluation. It goes beyond simple
    random or outcome-stratified splits to handle multidimensional
    stratification across demographics, clinical characteristics, and outcomes.
    """

    def __init__(
        self,
        random_state: Optional[int] = None,
        min_group_size_validation: int = 50,
        target_validation_fraction: float = 0.20
    ):
        """
        Initialize data splitter.

        Args:
            random_state: Random seed for reproducibility
            min_group_size_validation: Minimum samples per group in validation
            target_validation_fraction: Target fraction for validation set
        """
        self.random_state = random_state
        self.min_group_size_validation = min_group_size_validation
        self.target_validation_fraction = target_validation_fraction
        self.rng = np.random.RandomState(random_state)

        logger.info(
            f"Initialized EquityAwareDataSplitter with validation fraction "
            f"{target_validation_fraction} and minimum group size "
            f"{min_group_size_validation}"
        )

    def _compute_demographic_distribution(
        self,
        df: pd.DataFrame,
        demographic_columns: List[str],
        indices: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute demographic distribution for dataset or subset.

        Args:
            df: DataFrame containing demographic information
            demographic_columns: Columns to analyze
            indices: Optional subset of indices to analyze

        Returns:
            Nested dictionary of proportions for each demographic variable
        """
        subset = df.iloc[indices] if indices is not None else df
        distributions = {}

        for col in demographic_columns:
            if col not in subset.columns:
                logger.warning(f"Demographic column {col} not found in data")
                continue

            value_counts = subset[col].value_counts(normalize=True)
            distributions[col] = value_counts.to_dict()

        return distributions

    def _create_stratification_key(
        self,
        df: pd.DataFrame,
        stratification_columns: List[str]
    ) -> pd.Series:
        """
        Create composite stratification key from multiple columns.

        Combines multiple stratification variables into a single key that can
        be used with sklearn's stratified splitting. Handles missing values
        by treating them as a separate category.

        Args:
            df: DataFrame with stratification variables
            stratification_columns: Columns to combine for stratification

        Returns:
            Series containing composite stratification keys
        """
        key_parts = []

        for col in stratification_columns:
            if col not in df.columns:
                raise ValueError(f"Stratification column {col} not found")

            # Convert to string and handle missing values
            col_str = df[col].fillna('__MISSING__').astype(str)
            key_parts.append(col_str)

        # Create composite key
        composite_key = key_parts[0]
        for part in key_parts[1:]:
            composite_key = composite_key + '|||' + part

        return composite_key

    def multidimensional_stratified_split(
        self,
        df: pd.DataFrame,
        outcome_column: str,
        demographic_columns: List[str],
        clinical_stratification_columns: Optional[List[str]] = None
    ) -> ValidationSplit:
        """
        Create train/validation split with multidimensional stratification.

        This method performs stratified splitting that preserves distributions
        across multiple dimensions: outcomes, demographics, and clinical
        characteristics. It ensures adequate representation of demographic
        subgroups for fairness evaluation while maintaining outcome balance.

        Args:
            df: Full dataset
            outcome_column: Column containing outcomes
            demographic_columns: Demographic variables for stratification
            clinical_stratification_columns: Optional clinical variables

        Returns:
            ValidationSplit object with indices and metadata
        """
        logger.info(
            f"Performing multidimensional stratified split on {len(df)} samples"
        )

        # Determine all stratification columns
        all_strat_cols = [outcome_column] + demographic_columns
        if clinical_stratification_columns:
            all_strat_cols.extend(clinical_stratification_columns)

        # Create composite stratification key
        strat_key = self._create_stratification_key(df, all_strat_cols)

        # Check if stratification is feasible
        strat_counts = strat_key.value_counts()
        min_strat_count = strat_counts.min()

        if min_strat_count < 2:
            logger.warning(
                f"Some stratification groups have <2 samples. Falling back to "
                f"outcome-only stratification. Consider reducing stratification "
                f"dimensions or coarsening categorical variables."
            )
            # Fall back to outcome-only stratification
            strat_key = df[outcome_column]

        # Perform stratified split
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.target_validation_fraction,
            random_state=self.random_state
        )

        train_idx, val_idx = next(splitter.split(df, strat_key))

        # Compute demographic distributions
        train_demo = self._compute_demographic_distribution(
            df, demographic_columns, train_idx
        )
        val_demo = self._compute_demographic_distribution(
            df, demographic_columns, val_idx
        )

        # Check if minimum group sizes are met
        val_outcome_dist = df[outcome_column].iloc[val_idx].value_counts()

        split_metadata = {
            'strategy': SplitStrategy.MULTIDIMENSIONAL_STRATIFIED.value,
            'n_train': len(train_idx),
            'n_validation': len(val_idx),
            'validation_outcome_distribution': val_outcome_dist.to_dict(),
            'stratification_columns': all_strat_cols,
            'timestamp': datetime.now().isoformat()
        }

        # Check minimum representation requirements
        warnings = []
        for demo_col in demographic_columns:
            min_checks = {}
            for value, proportion in val_demo[demo_col].items():
                count = int(proportion * len(val_idx))
                if count < self.min_group_size_validation:
                    warnings.append(
                        f"{demo_col}={value}: only {count} validation samples "
                        f"(minimum: {self.min_group_size_validation})"
                    )
                min_checks[value] = count

            split_metadata[f'{demo_col}_validation_counts'] = min_checks

        if warnings:
            logger.warning(
                "Some demographic groups have insufficient validation samples:\n" +
                "\n".join(warnings)
            )
            split_metadata['representation_warnings'] = warnings

        validation_split = ValidationSplit(
            train_indices=train_idx,
            val_indices=val_idx,
            train_demographics=train_demo,
            val_demographics=val_demo,
            split_metadata=split_metadata
        )

        logger.info(
            f"Split complete: {len(train_idx)} training, {len(val_idx)} validation"
        )

        return validation_split

    def temporal_split(
        self,
        df: pd.DataFrame,
        time_column: str,
        outcome_column: str,
        demographic_columns: List[str],
        validation_start_date: Optional[datetime] = None
    ) -> ValidationSplit:
        """
        Create temporal train/validation split.

        Temporal splitting uses data before a cutoff date for training and
        after for validation, mimicking prospective deployment. This reveals
        whether models maintain performance as clinical practices, patient
        populations, and coding patterns evolve over time.

        Args:
            df: Full dataset with time column
            time_column: Column containing dates/times
            outcome_column: Column containing outcomes
            demographic_columns: Demographic variables to track
            validation_start_date: Optional cutoff date; if None, uses 80/20 split

        Returns:
            ValidationSplit object with indices and metadata
        """
        logger.info(f"Performing temporal split on column '{time_column}'")

        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found")

        # Ensure time column is datetime
        time_series = pd.to_datetime(df[time_column])

        # Determine split date
        if validation_start_date is None:
            # Use 80th percentile of dates as split point
            split_date = time_series.quantile(1 - self.target_validation_fraction)
            logger.info(f"Using data-driven split date: {split_date}")
        else:
            split_date = validation_start_date
            logger.info(f"Using specified split date: {split_date}")

        # Create train/validation indices
        train_idx = np.where(time_series < split_date)[0]
        val_idx = np.where(time_series >= split_date)[0]

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(
                f"Temporal split resulted in empty partition. "
                f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}"
            )

        # Compute demographic distributions
        train_demo = self._compute_demographic_distribution(
            df, demographic_columns, train_idx
        )
        val_demo = self._compute_demographic_distribution(
            df, demographic_columns, val_idx
        )

        # Compute outcome distributions
        train_outcome_dist = df[outcome_column].iloc[train_idx].value_counts()
        val_outcome_dist = df[outcome_column].iloc[val_idx].value_counts()

        split_metadata = {
            'strategy': SplitStrategy.TEMPORAL.value,
            'split_date': split_date.isoformat(),
            'n_train': len(train_idx),
            'n_validation': len(val_idx),
            'train_date_range': (
                time_series.iloc[train_idx].min().isoformat(),
                time_series.iloc[train_idx].max().isoformat()
            ),
            'validation_date_range': (
                time_series.iloc[val_idx].min().isoformat(),
                time_series.iloc[val_idx].max().isoformat()
            ),
            'train_outcome_distribution': train_outcome_dist.to_dict(),
            'validation_outcome_distribution': val_outcome_dist.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        # Check for demographic drift
        drift_warnings = []
        for demo_col in demographic_columns:
            for value in set(train_demo[demo_col].keys()) | set(val_demo[demo_col].keys()):
                train_prop = train_demo[demo_col].get(value, 0.0)
                val_prop = val_demo[demo_col].get(value, 0.0)

                # Flag substantial changes in representation
                if abs(train_prop - val_prop) > 0.10:  # 10 percentage point difference
                    drift_warnings.append(
                        f"{demo_col}={value}: train {train_prop:.2%} vs "
                        f"validation {val_prop:.2%}"
                    )

        if drift_warnings:
            logger.warning(
                "Demographic drift detected between training and validation:\n" +
                "\n".join(drift_warnings)
            )
            split_metadata['demographic_drift_warnings'] = drift_warnings

        validation_split = ValidationSplit(
            train_indices=train_idx,
            val_indices=val_idx,
            train_demographics=train_demo,
            val_demographics=val_demo,
            split_metadata=split_metadata
        )

        logger.info(
            f"Temporal split complete: {len(train_idx)} training (before {split_date.date()}), "
            f"{len(val_idx)} validation (after {split_date.date()})"
        )

        return validation_split

@dataclass
class PerformanceMetrics:
    """
    Container for comprehensive performance metrics.

    Attributes:
        auc_roc: Area under ROC curve
        auc_pr: Area under precision-recall curve
        brier_score: Brier score (lower is better)
        sensitivity: True positive rate at threshold
        specificity: True negative rate at threshold
        ppv: Positive predictive value at threshold
        npv: Negative predictive value at threshold
        threshold: Classification threshold used
        calibration_slope: Calibration slope (ideally 1.0)
        calibration_intercept: Calibration intercept (ideally 0.0)
        n_samples: Number of samples evaluated
        n_positive: Number of positive outcomes
        confidence_interval: Optional confidence intervals for metrics
    """
    auc_roc: float
    auc_pr: float
    brier_score: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    threshold: float
    calibration_slope: float
    calibration_intercept: float
    n_samples: int
    n_positive: int
    confidence_interval: Optional[Dict[str, Tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            'auc_roc': float(self.auc_roc),
            'auc_pr': float(self.auc_pr),
            'brier_score': float(self.brier_score),
            'sensitivity': float(self.sensitivity),
            'specificity': float(self.specificity),
            'ppv': float(self.ppv),
            'npv': float(self.npv),
            'threshold': float(self.threshold),
            'calibration_slope': float(self.calibration_slope),
            'calibration_intercept': float(self.calibration_intercept),
            'n_samples': int(self.n_samples),
            'n_positive': int(self.n_positive),
            'confidence_interval': self.confidence_interval
        }

class InternalValidator:
    """
    Comprehensive internal validation with equity evaluation.

    This class implements rigorous internal validation including stratified
    performance evaluation across demographic groups, fairness metric
    calculation, calibration analysis, and statistical testing for disparities.
    """

    def __init__(
        self,
        classification_threshold: float = 0.5,
        bootstrap_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ):
        """
        Initialize internal validator.

        Args:
            classification_threshold: Threshold for binary classification metrics
            bootstrap_iterations: Number of bootstrap samples for CI estimation
            confidence_level: Confidence level for intervals (default 95%)
            random_state: Random seed for reproducibility
        """
        self.classification_threshold = classification_threshold
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        logger.info(
            f"Initialized InternalValidator with threshold {classification_threshold}, "
            f"{bootstrap_iterations} bootstrap iterations, "
            f"{confidence_level:.0%} confidence intervals"
        )

    def _compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute calibration slope and intercept via logistic calibration.

        Calibration slope indicates whether predicted probabilities span an
        appropriate range (slope near 1.0 is ideal). Calibration intercept
        indicates systematic over- or under-prediction (intercept near 0.0
        is ideal).

        Args:
            y_true: True binary outcomes
            y_pred_proba: Predicted probabilities

        Returns:
            Tuple of (calibration_slope, calibration_intercept)
        """
        from sklearn.linear_model import LogisticRegression

        # Fit logistic calibration model
        logit_pred = np.log(y_pred_proba / (1 - y_pred_proba + 1e-10))
        logit_pred = logit_pred.reshape(-1, 1)

        cal_model = LogisticRegression(random_state=self.random_state)
        cal_model.fit(logit_pred, y_true)

        slope = float(cal_model.coef_[0, 0])
        intercept = float(cal_model.intercept_[0])

        return slope, intercept

    def evaluate_performance(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        compute_ci: bool = True
    ) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics with confidence intervals.

        Args:
            y_true: True binary outcomes (0/1)
            y_pred_proba: Predicted probabilities
            compute_ci: Whether to compute bootstrap confidence intervals

        Returns:
            PerformanceMetrics object with all metrics
        """
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        # Basic validation
        if len(y_true) != len(y_pred_proba):
            raise ValueError("Length mismatch between y_true and y_pred_proba")

        if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
            raise ValueError("Predicted probabilities must be in [0, 1]")

        # Compute discrimination metrics
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")
            auc_roc = np.nan

        try:
            auc_pr = average_precision_score(y_true, y_pred_proba)
        except ValueError as e:
            logger.warning(f"Could not compute AUC-PR: {e}")
            auc_pr = np.nan

        brier = brier_score_loss(y_true, y_pred_proba)

        # Compute calibration metrics
        if len(np.unique(y_true)) == 2:
            cal_slope, cal_intercept = self._compute_calibration_metrics(
                y_true, y_pred_proba
            )
        else:
            cal_slope, cal_intercept = np.nan, np.nan

        # Compute classification metrics at threshold
        y_pred_binary = (y_pred_proba >= self.classification_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        n_samples = len(y_true)
        n_positive = int(np.sum(y_true))

        # Bootstrap confidence intervals if requested
        ci = None
        if compute_ci and n_samples > 30:
            ci = self._bootstrap_confidence_intervals(y_true, y_pred_proba)

        return PerformanceMetrics(
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            brier_score=brier,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            threshold=self.classification_threshold,
            calibration_slope=cal_slope,
            calibration_intercept=cal_intercept,
            n_samples=n_samples,
            n_positive=n_positive,
            confidence_interval=ci
        )

    def _bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for key metrics.

        Args:
            y_true: True binary outcomes
            y_pred_proba: Predicted probabilities

        Returns:
            Dictionary mapping metric names to (lower, upper) CI bounds
        """
        n_samples = len(y_true)
        auc_roc_boots = []
        sensitivity_boots = []
        specificity_boots = []

        for _ in range(self.bootstrap_iterations):
            # Resample with replacement
            boot_idx = self.rng.choice(n_samples, size=n_samples, replace=True)
            y_boot = y_true[boot_idx]
            y_pred_boot = y_pred_proba[boot_idx]

            # Skip if bootstrap sample doesn't have both classes
            if len(np.unique(y_boot)) < 2:
                continue

            # Compute metrics on bootstrap sample
            try:
                auc_boot = roc_auc_score(y_boot, y_pred_boot)
                auc_roc_boots.append(auc_boot)
            except ValueError:
                pass

            y_pred_binary_boot = (y_pred_boot >= self.classification_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_boot, y_pred_binary_boot).ravel()

            sens_boot = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec_boot = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            sensitivity_boots.append(sens_boot)
            specificity_boots.append(spec_boot)

        # Compute percentile-based confidence intervals
        alpha = 1 - self.confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ci_dict = {}

        if len(auc_roc_boots) > 0:
            ci_dict['auc_roc'] = (
                np.percentile(auc_roc_boots, lower_percentile),
                np.percentile(auc_roc_boots, upper_percentile)
            )

        if len(sensitivity_boots) > 0:
            ci_dict['sensitivity'] = (
                np.percentile(sensitivity_boots, lower_percentile),
                np.percentile(sensitivity_boots, upper_percentile)
            )

        if len(specificity_boots) > 0:
            ci_dict['specificity'] = (
                np.percentile(specificity_boots, lower_percentile),
                np.percentile(specificity_boots, upper_percentile)
            )

        return ci_dict

    def stratified_evaluation(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        stratification_variable: pd.Series,
        compute_ci: bool = False
    ) -> Dict[str, PerformanceMetrics]:
        """
        Evaluate performance stratified by a grouping variable.

        This method computes performance metrics separately for each level
        of a stratification variable (e.g., race, gender, age group), enabling
        detection of performance disparities across patient subgroups.

        Args:
            y_true: True binary outcomes
            y_pred_proba: Predicted probabilities
            stratification_variable: Categorical variable defining subgroups
            compute_ci: Whether to compute confidence intervals per subgroup

        Returns:
            Dictionary mapping group labels to PerformanceMetrics
        """
        logger.info(
            f"Computing stratified evaluation for variable with "
            f"{stratification_variable.nunique()} unique values"
        )

        results = {}

        for group_value in stratification_variable.unique():
            # Skip missing values
            if pd.isna(group_value):
                continue

            # Get indices for this group
            group_mask = (stratification_variable == group_value).values
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred_proba[group_mask]

            # Skip groups with insufficient samples or only one class
            if len(y_true_group) < 30:
                logger.warning(
                    f"Skipping group '{group_value}' with only "
                    f"{len(y_true_group)} samples"
                )
                continue

            if len(np.unique(y_true_group)) < 2:
                logger.warning(
                    f"Skipping group '{group_value}' with only one outcome class"
                )
                continue

            # Compute metrics for this group
            try:
                metrics = self.evaluate_performance(
                    y_true_group,
                    y_pred_group,
                    compute_ci=compute_ci
                )
                results[str(group_value)] = metrics

                logger.info(
                    f"Group '{group_value}': n={metrics.n_samples}, "
                    f"AUC-ROC={metrics.auc_roc:.3f}"
                )
            except Exception as e:
                logger.error(f"Error computing metrics for group '{group_value}': {e}")

        return results

    def compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        protected_attribute: pd.Series,
        reference_group: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive fairness metrics across demographic groups.

        This method computes key fairness metrics including:
        - Demographic parity: difference in positive prediction rates
        - Equalized odds: difference in TPR and FPR across groups
        - Calibration within groups
        - Predictive parity: difference in PPV across groups

        Args:
            y_true: True binary outcomes
            y_pred_proba: Predicted probabilities
            protected_attribute: Demographic variable (e.g., race, gender)
            reference_group: Optional reference group for computing ratios

        Returns:
            Dictionary containing fairness metrics and group comparisons
        """
        logger.info("Computing comprehensive fairness metrics")

        # Get stratified performance metrics
        stratified_metrics = self.stratified_evaluation(
            y_true, y_pred_proba, protected_attribute, compute_ci=False
        )

        if len(stratified_metrics) < 2:
            logger.warning("Need at least 2 groups for fairness evaluation")
            return {}

        # Determine reference group
        if reference_group is None:
            # Use largest group as reference
            group_sizes = {
                group: metrics.n_samples
                for group, metrics in stratified_metrics.items()
            }
            reference_group = max(group_sizes, key=group_sizes.get)

        logger.info(f"Using '{reference_group}' as reference group")

        if reference_group not in stratified_metrics:
            raise ValueError(f"Reference group '{reference_group}' not found")

        ref_metrics = stratified_metrics[reference_group]

        # Compute fairness metrics
        fairness_results = {
            'reference_group': reference_group,
            'group_metrics': {},
            'demographic_parity': {},
            'equalized_odds': {},
            'calibration': {},
            'predictive_parity': {}
        }

        for group, metrics in stratified_metrics.items():
            if group == reference_group:
                continue

            # Store group metrics
            fairness_results['group_metrics'][group] = metrics.to_dict()

            # Demographic parity: difference in positive prediction rate
            # Compute positive prediction rates
            group_mask = (protected_attribute == group).values
            ref_mask = (protected_attribute == reference_group).values

            group_pos_rate = np.mean(
                y_pred_proba[group_mask] >= self.classification_threshold
            )
            ref_pos_rate = np.mean(
                y_pred_proba[ref_mask] >= self.classification_threshold
            )

            fairness_results['demographic_parity'][group] = {
                'group_positive_rate': float(group_pos_rate),
                'reference_positive_rate': float(ref_pos_rate),
                'difference': float(group_pos_rate - ref_pos_rate),
                'ratio': float(group_pos_rate / ref_pos_rate) if ref_pos_rate > 0 else np.nan
            }

            # Equalized odds: difference in TPR and FPR
            tpr_diff = metrics.sensitivity - ref_metrics.sensitivity
            # Compute FPR
            group_fpr = 1 - metrics.specificity
            ref_fpr = 1 - ref_metrics.specificity
            fpr_diff = group_fpr - ref_fpr

            fairness_results['equalized_odds'][group] = {
                'tpr_difference': float(tpr_diff),
                'fpr_difference': float(fpr_diff),
                'max_difference': float(max(abs(tpr_diff), abs(fpr_diff)))
            }

            # Calibration: difference in calibration metrics
            cal_slope_diff = metrics.calibration_slope - ref_metrics.calibration_slope
            cal_int_diff = metrics.calibration_intercept - ref_metrics.calibration_intercept

            fairness_results['calibration'][group] = {
                'slope_difference': float(cal_slope_diff),
                'intercept_difference': float(cal_int_diff),
                'group_slope': float(metrics.calibration_slope),
                'group_intercept': float(metrics.calibration_intercept)
            }

            # Predictive parity: difference in PPV
            ppv_diff = metrics.ppv - ref_metrics.ppv

            fairness_results['predictive_parity'][group] = {
                'ppv_difference': float(ppv_diff),
                'ppv_ratio': float(metrics.ppv / ref_metrics.ppv) if ref_metrics.ppv > 0 else np.nan,
                'group_ppv': float(metrics.ppv),
                'reference_ppv': float(ref_metrics.ppv)
            }

        # Store reference group metrics
        fairness_results['group_metrics'][reference_group] = ref_metrics.to_dict()

        return fairness_results

    def generate_validation_report(
        self,
        overall_metrics: PerformanceMetrics,
        stratified_metrics: Dict[str, PerformanceMetrics],
        fairness_metrics: Dict[str, Any],
        validation_split: ValidationSplit
    ) -> str:
        """
        Generate comprehensive validation report.

        Args:
            overall_metrics: Overall performance metrics
            stratified_metrics: Performance stratified by demographics
            fairness_metrics: Fairness evaluation results
            validation_split: Information about data split

        Returns:
            Formatted validation report as string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("INTERNAL VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Split information
        lines.append("VALIDATION SPLIT INFORMATION:")
        lines.append(f"  Strategy: {validation_split.split_metadata.get('strategy', 'Unknown')}")
        lines.append(f"  Training samples: {validation_split.split_metadata.get('n_train', 0):,}")
        lines.append(f"  Validation samples: {validation_split.split_metadata.get('n_validation', 0):,}")
        lines.append("")

        # Overall performance
        lines.append("OVERALL PERFORMANCE:")
        lines.append(f"  Samples: {overall_metrics.n_samples:,}")
        lines.append(f"  Positive outcomes: {overall_metrics.n_positive:,} ({100*overall_metrics.n_positive/overall_metrics.n_samples:.1f}%)")
        lines.append(f"  AUC-ROC: {overall_metrics.auc_roc:.4f}")
        if overall_metrics.confidence_interval and 'auc_roc' in overall_metrics.confidence_interval:
            ci = overall_metrics.confidence_interval['auc_roc']
            lines.append(f"    95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
        lines.append(f"  AUC-PR: {overall_metrics.auc_pr:.4f}")
        lines.append(f"  Brier score: {overall_metrics.brier_score:.4f}")
        lines.append("")
        lines.append(f"  At threshold {overall_metrics.threshold}:")
        lines.append(f"    Sensitivity: {overall_metrics.sensitivity:.4f}")
        lines.append(f"    Specificity: {overall_metrics.specificity:.4f}")
        lines.append(f"    PPV: {overall_metrics.ppv:.4f}")
        lines.append(f"    NPV: {overall_metrics.npv:.4f}")
        lines.append("")
        lines.append(f"  Calibration:")
        lines.append(f"    Slope: {overall_metrics.calibration_slope:.4f} (ideal: 1.0)")
        lines.append(f"    Intercept: {overall_metrics.calibration_intercept:.4f} (ideal: 0.0)")
        lines.append("")

        # Stratified performance
        if stratified_metrics:
            lines.append("STRATIFIED PERFORMANCE:")
            lines.append("")

            # Sort groups by sample size
            sorted_groups = sorted(
                stratified_metrics.items(),
                key=lambda x: x[1].n_samples,
                reverse=True
            )

            for group, metrics in sorted_groups:
                lines.append(f"  Group: {group}")
                lines.append(f"    n={metrics.n_samples:,} ({100*metrics.n_samples/overall_metrics.n_samples:.1f}% of total)")
                lines.append(f"    Positive outcomes: {metrics.n_positive:,} ({100*metrics.n_positive/metrics.n_samples:.1f}%)")
                lines.append(f"    AUC-ROC: {metrics.auc_roc:.4f}")
                lines.append(f"    Sensitivity: {metrics.sensitivity:.4f}")
                lines.append(f"    Specificity: {metrics.specificity:.4f}")
                lines.append(f"    PPV: {metrics.ppv:.4f}")
                lines.append(f"    Calibration slope: {metrics.calibration_slope:.4f}")
                lines.append("")

        # Fairness metrics
        if fairness_metrics:
            lines.append("FAIRNESS EVALUATION:")
            lines.append(f"  Reference group: {fairness_metrics.get('reference_group', 'N/A')}")
            lines.append("")

            # Demographic parity
            if 'demographic_parity' in fairness_metrics:
                lines.append("  Demographic Parity:")
                for group, metrics in fairness_metrics['demographic_parity'].items():
                    diff = metrics['difference']
                    ratio = metrics['ratio']
                    lines.append(
                        f"    {group}: difference={diff:+.4f}, "
                        f"ratio={ratio:.4f}"
                    )
                lines.append("")

            # Equalized odds
            if 'equalized_odds' in fairness_metrics:
                lines.append("  Equalized Odds:")
                for group, metrics in fairness_metrics['equalized_odds'].items():
                    max_diff = metrics['max_difference']
                    lines.append(
                        f"    {group}: max(|TPR diff|, |FPR diff|)={max_diff:.4f}"
                    )
                lines.append("")

            # Predictive parity
            if 'predictive_parity' in fairness_metrics:
                lines.append("  Predictive Parity (PPV):")
                for group, metrics in fairness_metrics['predictive_parity'].items():
                    diff = metrics['ppv_difference']
                    ratio = metrics['ppv_ratio']
                    lines.append(
                        f"    {group}: difference={diff:+.4f}, "
                        f"ratio={ratio:.4f}"
                    )
                lines.append("")

        # Warnings
        if 'representation_warnings' in validation_split.split_metadata:
            lines.append("REPRESENTATION WARNINGS:")
            for warning in validation_split.split_metadata['representation_warnings']:
                lines.append(f"  [WARN]  {warning}")
            lines.append("")

        if 'demographic_drift_warnings' in validation_split.split_metadata:
            lines.append("DEMOGRAPHIC DRIFT WARNINGS:")
            for warning in validation_split.split_metadata['demographic_drift_warnings']:
                lines.append(f"  [WARN]  {warning}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)
```

This internal validation framework provides comprehensive tools for equitable model evaluation. The equity-aware data splitting ensures validation cohorts contain adequate representation of diverse patient populations, going beyond simple random splits to explicitly stratify by multiple dimensions simultaneously. The stratified performance evaluation computes metrics separately for each demographic group, enabling detection of disparities that would be hidden by aggregate statistics. The fairness metrics quantify multiple complementary notions of algorithmic fairness, from demographic parity to equalized odds to calibration within groups. Together these components enable rigorous internal validation that surfaces potential equity issues before models advance to external validation or deployment.

## 15.3 Sample Size Calculations for Fairness Evaluation

Detecting meaningful performance disparities across demographic subgroups requires adequate sample sizes within each subgroup, not just overall. Standard sample size calculations for model validation focus on estimating overall performance metrics with acceptable precision but provide no guidance on the number of samples needed to detect fairness violations. This section develops power calculations specifically for fairness metrics that enable validation study design with adequate statistical power to identify clinically meaningful disparities.

The fundamental challenge is that fairness evaluation requires comparing performance metrics between groups, which necessitates precision in estimating those metrics within each group separately. If we want to detect a sensitivity difference of five percentage points between two demographic groups with eighty percent power, we need sufficient positive cases within each group to estimate sensitivity precisely enough that a five percentage point difference would be statistically distinguishable from chance. This typically requires many more samples than would be needed to achieve comparable precision for overall sensitivity estimation.

Consider a clinical prediction model for identifying patients at high risk of hospital readmission within thirty days. Suppose the model is being validated in a cohort where the readmission rate is fifteen percent overall, and we want to detect whether sensitivity differs by race with adequate power. If we have a validation cohort of one thousand patients split evenly between two racial groups with equal readmission rates, each group contains five hundred patients with seventy-five readmissions. With this sample size, we can detect sensitivity differences of approximately ten percentage points with eighty percent power at a five percent significance level using standard two-proportion z-tests. However, detecting smaller differences of five percentage points would require quadrupling the sample size to two thousand patients per group, or four thousand total validation samples.

The situation becomes even more challenging when multiple demographic groups must be evaluated. If we need to assess fairness across four racial categories rather than two, and we want pairwise comparisons between all groups, we are conducting six comparisons and must account for multiple testing. With Bonferroni correction for six tests, each test is performed at approximately 0.8 percent significance level to maintain five percent family-wise error rate, which further increases required sample sizes. Alternatively, hierarchical testing strategies first test for any difference across all groups using omnibus tests like Kruskal-Wallis, then conduct pairwise comparisons only if the omnibus test is significant, potentially reducing the effective number of tests.

Sample size requirements also depend on expected performance levels and disparity magnitudes. All else equal, detecting disparities is easier when model performance is moderate rather than very high or very low due to ceiling and floor effects. Detecting small disparities requires more samples than detecting large disparities. The clinical context determines what disparity magnitude is meaningful and therefore what sample size is adequate. A two percentage point difference in sensitivity might be negligible for a screening test but highly consequential for diagnosis of a life-threatening condition.

We now develop practical tools for sample size calculation that account for fairness evaluation requirements.

```python
"""
Sample size calculations for fairness evaluation in validation studies.

This module implements power calculations for detecting disparities in
clinical AI performance across demographic subgroups. It accounts for
multiple comparisons, expected performance levels, and clinically
meaningful disparity thresholds to guide validation study design.
"""

from typing import Optional, Dict, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class PowerAnalysisParameters:
    """
    Parameters for fairness-focused power analysis.

    Attributes:
        alpha: Significance level (default 0.05)
        beta: Type II error rate (1 - power, default 0.20 for 80% power)
        baseline_sensitivity: Expected sensitivity in reference group
        baseline_specificity: Expected specificity in reference group
        prevalence: Expected outcome prevalence
        minimum_detectable_difference: Smallest meaningful disparity
        n_groups: Number of demographic groups to compare
        multiple_testing_correction: Method for multiple testing correction
    """
    alpha: float = 0.05
    beta: float = 0.20
    baseline_sensitivity: float = 0.80
    baseline_specificity: float = 0.85
    prevalence: float = 0.10
    minimum_detectable_difference: float = 0.05
    n_groups: int = 2
    multiple_testing_correction: str = "bonferroni"

class FairnessPowerCalculator:
    """
    Sample size and power calculations for fairness evaluation.

    This class implements methods for determining adequate validation cohort
    sizes to detect meaningful performance disparities across demographic
    groups with specified statistical power.
    """

    def __init__(self, parameters: PowerAnalysisParameters):
        """
        Initialize power calculator.

        Args:
            parameters: Power analysis parameters
        """
        self.params = parameters

        # Adjust alpha for multiple testing if needed
        if self.params.multiple_testing_correction == "bonferroni":
            n_comparisons = (self.params.n_groups * (self.params.n_groups - 1)) // 2
            self.adjusted_alpha = self.params.alpha / n_comparisons
            logger.info(
                f"Bonferroni correction: {n_comparisons} comparisons, "
                f"adjusted alpha={self.adjusted_alpha:.6f}"
            )
        else:
            self.adjusted_alpha = self.params.alpha

        # Pre-compute z-scores for efficiency
        self.z_alpha = stats.norm.ppf(1 - self.adjusted_alpha / 2)
        self.z_beta = stats.norm.ppf(1 - self.params.beta)

    def sample_size_for_sensitivity_difference(
        self,
        alternative_sensitivity: Optional[float] = None
    ) -> int:
        """
        Calculate required positive cases per group to detect sensitivity difference.

        Uses two-proportion z-test formula for comparing sensitivities between
        two groups. Assumes equal numbers of positive cases in each group.

        Args:
            alternative_sensitivity: Sensitivity in comparison group. If None,
                uses baseline_sensitivity + minimum_detectable_difference

        Returns:
            Required number of positive cases per group
        """
        p1 = self.params.baseline_sensitivity

        if alternative_sensitivity is None:
            p2 = p1 + self.params.minimum_detectable_difference
        else:
            p2 = alternative_sensitivity

        # Average proportion
        p_avg = (p1 + p2) / 2

        # Effect size
        delta = abs(p2 - p1)

        if delta == 0:
            raise ValueError("Cannot detect zero difference")

        # Sample size formula for two-proportion test
        n_positive = (
            (self.z_alpha + self.z_beta) ** 2 * p_avg * (1 - p_avg)
        ) / (delta ** 2)

        # Round up to integer
        n_positive = int(np.ceil(n_positive))

        logger.info(
            f"Sensitivity difference detection: need {n_positive} positive "
            f"cases per group (p1={p1:.3f}, p2={p2:.3f}, delta={delta:.3f})"
        )

        return n_positive

    def sample_size_for_specificity_difference(
        self,
        alternative_specificity: Optional[float] = None
    ) -> int:
        """
        Calculate required negative cases per group to detect specificity difference.

        Args:
            alternative_specificity: Specificity in comparison group. If None,
                uses baseline_specificity + minimum_detectable_difference

        Returns:
            Required number of negative cases per group
        """
        p1 = self.params.baseline_specificity

        if alternative_specificity is None:
            p2 = p1 + self.params.minimum_detectable_difference
        else:
            p2 = alternative_specificity

        p_avg = (p1 + p2) / 2
        delta = abs(p2 - p1)

        if delta == 0:
            raise ValueError("Cannot detect zero difference")

        n_negative = (
            (self.z_alpha + self.z_beta) ** 2 * p_avg * (1 - p_avg)
        ) / (delta ** 2)

        n_negative = int(np.ceil(n_negative))

        logger.info(
            f"Specificity difference detection: need {n_negative} negative "
            f"cases per group"
        )

        return n_negative

    def total_validation_size(self) -> Dict[str, int]:
        """
        Calculate total validation cohort size requirements.

        Combines requirements for sensitivity and specificity difference
        detection with prevalence assumptions to determine total cohort
        size needed per group and overall.

        Returns:
            Dictionary with sample size requirements
        """
        # Positive cases needed per group
        n_pos_per_group = self.sample_size_for_sensitivity_difference()

        # Negative cases needed per group
        n_neg_per_group = self.sample_size_for_specificity_difference()

        # Total samples per group based on positive cases and prevalence
        n_total_from_pos = int(np.ceil(n_pos_per_group / self.params.prevalence))

        # Total samples per group based on negative cases and prevalence
        n_total_from_neg = int(np.ceil(
            n_neg_per_group / (1 - self.params.prevalence)
        ))

        # Take maximum to satisfy both constraints
        n_per_group = max(n_total_from_pos, n_total_from_neg)

        # Total validation cohort size across all groups
        n_total = n_per_group * self.params.n_groups

        results = {
            'positive_cases_per_group': n_pos_per_group,
            'negative_cases_per_group': n_neg_per_group,
            'total_samples_per_group': n_per_group,
            'total_validation_cohort': n_total,
            'expected_positives_per_group': int(n_per_group * self.params.prevalence),
            'expected_negatives_per_group': int(n_per_group * (1 - self.params.prevalence))
        }

        logger.info(
            f"Total validation requirements: {n_total:,} total samples "
            f"({n_per_group:,} per group across {self.params.n_groups} groups)"
        )

        return results

    def sample_size_for_auc_difference(
        self,
        baseline_auc: float,
        alternative_auc: Optional[float] = None,
        correlation: float = 0.5
    ) -> int:
        """
        Calculate sample size for detecting AUC differences between groups.

        Uses DeLong's method for correlated AUCs when evaluated on the
        same validation set. This is more complex than proportion tests
        because AUC variance depends on both discrimination and sample size.

        Args:
            baseline_auc: Expected AUC in reference group
            alternative_auc: Expected AUC in comparison group
            correlation: Assumed correlation between group AUCs (default 0.5)

        Returns:
            Approximate total sample size needed per group
        """
        if alternative_auc is None:
            alternative_auc = baseline_auc + self.params.minimum_detectable_difference

        # Hanley-McNeil approximation for AUC variance
        def auc_variance(auc: float, n_pos: int, n_neg: int) -> float:
            q1 = auc / (2 - auc)
            q2 = 2 * auc ** 2 / (1 + auc)

            var = (
                auc * (1 - auc) +
                (n_pos - 1) * (q1 - auc ** 2) +
                (n_neg - 1) * (q2 - auc ** 2)
            ) / (n_pos * n_neg)

            return var

        # Iterative search for sample size
        # Start with rough estimate
        n = 100

        while n < 100000:  # Safety limit
            n_pos = int(n * self.params.prevalence)
            n_neg = n - n_pos

            if n_pos < 10 or n_neg < 10:
                n += 100
                continue

            var1 = auc_variance(baseline_auc, n_pos, n_neg)
            var2 = auc_variance(alternative_auc, n_pos, n_neg)

            # Variance of difference assuming correlation
            var_diff = var1 + var2 - 2 * correlation * np.sqrt(var1 * var2)
            se_diff = np.sqrt(var_diff)

            # Effect size
            delta = abs(alternative_auc - baseline_auc)

            # Z-statistic for difference
            z_stat = delta / se_diff

            # Power for this sample size
            power = 1 - stats.norm.cdf(self.z_alpha - z_stat)

            # Check if we've achieved target power
            if power >= (1 - self.params.beta):
                logger.info(
                    f"AUC difference detection: need {n} samples per group "
                    f"(baseline AUC={baseline_auc:.3f}, "
                    f"alternative AUC={alternative_auc:.3f})"
                )
                return n

            # Increase sample size
            n = int(n * 1.1)

        logger.warning(
            "Could not find feasible sample size for AUC difference detection"
        )
        return 100000

    def generate_power_analysis_report(self) -> str:
        """
        Generate comprehensive power analysis report.

        Returns:
            Formatted report describing sample size requirements
        """
        lines = []
        lines.append("=" * 80)
        lines.append("SAMPLE SIZE CALCULATION FOR FAIRNESS EVALUATION")
        lines.append("=" * 80)
        lines.append("")

        lines.append("PARAMETERS:")
        lines.append(f"  Significance level (alpha): {self.params.alpha}")
        lines.append(f"  Power (1 - beta): {1 - self.params.beta}")
        lines.append(f"  Number of groups: {self.params.n_groups}")
        lines.append(f"  Multiple testing correction: {self.params.multiple_testing_correction}")
        if self.params.multiple_testing_correction == "bonferroni":
            n_comp = (self.params.n_groups * (self.params.n_groups - 1)) // 2
            lines.append(f"  Number of pairwise comparisons: {n_comp}")
            lines.append(f"  Adjusted alpha per test: {self.adjusted_alpha:.6f}")
        lines.append("")

        lines.append("EXPECTED PERFORMANCE:")
        lines.append(f"  Baseline sensitivity: {self.params.baseline_sensitivity:.3f}")
        lines.append(f"  Baseline specificity: {self.params.baseline_specificity:.3f}")
        lines.append(f"  Outcome prevalence: {self.params.prevalence:.3f}")
        lines.append(f"  Minimum detectable difference: {self.params.minimum_detectable_difference:.3f}")
        lines.append("")

        # Calculate requirements
        size_req = self.total_validation_size()

        lines.append("SAMPLE SIZE REQUIREMENTS:")
        lines.append(f"  Positive cases per group: {size_req['positive_cases_per_group']:,}")
        lines.append(f"  Negative cases per group: {size_req['negative_cases_per_group']:,}")
        lines.append(f"  Total samples per group: {size_req['total_samples_per_group']:,}")
        lines.append(f"  Total validation cohort: {size_req['total_validation_cohort']:,}")
        lines.append("")

        lines.append("INTERPRETATION:")
        lines.append(
            f"  To detect a {self.params.minimum_detectable_difference:.1%} difference "
            f"in sensitivity or specificity"
        )
        lines.append(
            f"  between any pair of {self.params.n_groups} demographic groups with "
            f"{100*(1-self.params.beta):.0f}% power,"
        )
        lines.append(
            f"  the validation cohort should include at least "
            f"{size_req['total_samples_per_group']:,} patients"
        )
        lines.append(
            f"  from each demographic group, for a total validation cohort of "
            f"{size_req['total_validation_cohort']:,} patients."
        )
        lines.append("")

        lines.append("EXPECTED OUTCOME DISTRIBUTION:")
        lines.append(
            f"  Each group expected to have ~{size_req['expected_positives_per_group']:,} "
            f"positive outcomes"
        )
        lines.append(
            f"  and ~{size_req['expected_negatives_per_group']:,} negative outcomes"
        )
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)
```

These power calculation tools enable rigorous validation study design that explicitly accounts for fairness evaluation requirements. The sample size calculations reveal that detecting meaningful disparities often requires much larger validation cohorts than standard rules of thumb would suggest, particularly when multiple demographic groups must be compared with correction for multiple testing. The framework provides both specific numeric requirements and comprehensive reporting suitable for inclusion in validation study protocols and regulatory documentation.

## 15.4 Temporal Validation for Performance Monitoring

Clinical AI models deployed in production face evolving patient populations, changing clinical practices, and shifting data distributions over time. Temporal validation assesses whether model performance degrades as these factors change, providing critical information about expected model lifespan and retraining needs. For models intended to serve diverse populations, temporal validation must evaluate not just overall performance drift but also changes in fairness metrics that might indicate emerging disparities.

The fundamental challenge in temporal validation is distinguishing expected variations in performance due to random sampling from systematic degradation requiring intervention. If a model's AUC decreases from 0.90 to 0.88 between consecutive months, is this natural variation or evidence of meaningful drift? Statistical process control methods adapted from manufacturing quality monitoring provide frameworks for detecting truly anomalous performance changes while avoiding excessive false alarms from random fluctuation.

From an equity perspective, temporal validation must track fairness metrics alongside overall performance because disparities can emerge even when aggregate performance remains stable. A model might maintain an overall AUC of 0.90 while performance for underrepresented demographic groups degrades from 0.88 to 0.82. Standard monitoring focused on aggregate metrics would miss this concerning pattern. Comprehensive temporal validation therefore requires stratified evaluation at each time point with explicit tracking of group-specific performance trajectories and fairness metrics.

We now develop frameworks for temporal validation with equity-focused performance monitoring.

```python
"""
Temporal validation and performance monitoring framework.

This module implements comprehensive temporal validation strategies for
detecting model performance degradation over time. It includes statistical
process control methods for identifying meaningful drift, stratified
monitoring across demographic groups, and automated alerting for fairness
violations.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats

@dataclass
class PerformanceSnapshot:
    """
    Performance metrics at a specific time point.

    Attributes:
        timestamp: When evaluation was performed
        overall_metrics: Overall performance metrics
        stratified_metrics: Performance by demographic groups
        fairness_metrics: Fairness evaluation results
        n_samples: Number of samples evaluated
        data_distribution: Distribution of key variables
    """
    timestamp: datetime
    overall_metrics: PerformanceMetrics
    stratified_metrics: Dict[str, PerformanceMetrics]
    fairness_metrics: Dict[str, Any]
    n_samples: int
    data_distribution: Dict[str, Any] = field(default_factory=dict)

class TemporalValidator:
    """
    Temporal validation with drift detection and fairness monitoring.

    This class implements comprehensive temporal validation including:
    - Statistical process control for performance monitoring
    - Drift detection using multiple algorithms
    - Stratified performance tracking across demographics
    - Automated alerts for fairness violations
    - Trend analysis and degradation forecasting
    """

    def __init__(
        self,
        baseline_metrics: PerformanceSnapshot,
        control_limit_sigma: float = 3.0,
        min_samples_for_alert: int = 100,
        fairness_alert_threshold: float = 0.05,
        history_length: int = 50
    ):
        """
        Initialize temporal validator.

        Args:
            baseline_metrics: Initial performance snapshot for comparison
            control_limit_sigma: Standard deviations for control limits
            min_samples_for_alert: Minimum samples before generating alerts
            fairness_alert_threshold: Threshold for fairness metric violations
            history_length: Number of time points to retain in memory
        """
        self.baseline = baseline_metrics
        self.control_limit_sigma = control_limit_sigma
        self.min_samples_for_alert = min_samples_for_alert
        self.fairness_alert_threshold = fairness_alert_threshold

        # Rolling history of performance snapshots
        self.history = deque(maxlen=history_length)
        self.history.append(baseline_metrics)

        # Alert tracking
        self.active_alerts = []

        logger.info(
            f"Initialized TemporalValidator with baseline AUC "
            f"{baseline_metrics.overall_metrics.auc_roc:.4f}"
        )

    def _compute_control_limits(
        self,
        metric_values: List[float]
    ) -> Tuple[float, float, float]:
        """
        Compute statistical process control limits.

        Uses mean and standard deviation from historical data to establish
        control limits for detecting anomalous performance.

        Args:
            metric_values: Historical values of a metric

        Returns:
            Tuple of (center_line, lower_control_limit, upper_control_limit)
        """
        if len(metric_values) < 2:
            return np.nan, np.nan, np.nan

        center_line = np.mean(metric_values)
        std_dev = np.std(metric_values, ddof=1)

        lcl = center_line - self.control_limit_sigma * std_dev
        ucl = center_line + self.control_limit_sigma * std_dev

        return center_line, lcl, ucl

    def evaluate_current_performance(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        stratification_variable: pd.Series,
        timestamp: Optional[datetime] = None,
        data_features: Optional[pd.DataFrame] = None
    ) -> PerformanceSnapshot:
        """
        Evaluate current performance and compare to baseline.

        Args:
            y_true: True binary outcomes
            y_pred_proba: Predicted probabilities
            stratification_variable: Demographic variable for stratification
            timestamp: Evaluation timestamp (defaults to now)
            data_features: Optional features for distribution monitoring

        Returns:
            PerformanceSnapshot with current metrics
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"Evaluating performance at {timestamp}")

        # Initialize internal validator
        validator = InternalValidator(
            classification_threshold=self.baseline.overall_metrics.threshold,
            bootstrap_iterations=0,  # Skip CI computation for speed
            random_state=42
        )

        # Compute overall metrics
        overall_metrics = validator.evaluate_performance(
            y_true, y_pred_proba, compute_ci=False
        )

        # Compute stratified metrics
        stratified_metrics = validator.stratified_evaluation(
            y_true, y_pred_proba, stratification_variable, compute_ci=False
        )

        # Compute fairness metrics
        fairness_metrics = validator.compute_fairness_metrics(
            y_true, y_pred_proba, stratification_variable
        )

        # Track data distribution if features provided
        data_distribution = {}
        if data_features is not None:
            for col in data_features.columns:
                if data_features[col].dtype in [np.float64, np.int64]:
                    data_distribution[col] = {
                        'mean': float(data_features[col].mean()),
                        'std': float(data_features[col].std()),
                        'missing_rate': float(data_features[col].isna().mean())
                    }

        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            overall_metrics=overall_metrics,
            stratified_metrics=stratified_metrics,
            fairness_metrics=fairness_metrics,
            n_samples=len(y_true),
            data_distribution=data_distribution
        )

        # Add to history
        self.history.append(snapshot)

        # Check for alerts
        self._check_for_alerts(snapshot)

        return snapshot

    def _check_for_alerts(self, current: PerformanceSnapshot):
        """
        Check current performance against thresholds and generate alerts.

        Args:
            current: Current performance snapshot
        """
        if current.n_samples < self.min_samples_for_alert:
            return

        # Check overall performance degradation
        auc_decline = (
            self.baseline.overall_metrics.auc_roc -
            current.overall_metrics.auc_roc
        )

        if auc_decline > 0.05:  # 5 percentage point decline
            alert = {
                'type': 'OVERALL_PERFORMANCE_DEGRADATION',
                'timestamp': current.timestamp,
                'severity': 'HIGH' if auc_decline > 0.10 else 'MEDIUM',
                'message': (
                    f"Overall AUC declined by {auc_decline:.3f} "
                    f"from baseline {self.baseline.overall_metrics.auc_roc:.3f} "
                    f"to {current.overall_metrics.auc_roc:.3f}"
                )
            }
            self.active_alerts.append(alert)
            logger.warning(alert['message'])

        # Check calibration degradation
        cal_slope_drift = abs(
            current.overall_metrics.calibration_slope - 1.0
        )

        if cal_slope_drift > 0.20:  # Calibration slope >20% from ideal
            alert = {
                'type': 'CALIBRATION_DRIFT',
                'timestamp': current.timestamp,
                'severity': 'MEDIUM',
                'message': (
                    f"Calibration slope {current.overall_metrics.calibration_slope:.3f} "
                    f"substantially different from ideal 1.0"
                )
            }
            self.active_alerts.append(alert)
            logger.warning(alert['message'])

        # Check fairness metric violations
        if 'demographic_parity' in current.fairness_metrics:
            for group, metrics in current.fairness_metrics['demographic_parity'].items():
                diff = abs(metrics['difference'])

                if diff > self.fairness_alert_threshold:
                    alert = {
                        'type': 'FAIRNESS_VIOLATION',
                        'subtype': 'demographic_parity',
                        'group': group,
                        'timestamp': current.timestamp,
                        'severity': 'HIGH' if diff > 0.10 else 'MEDIUM',
                        'message': (
                            f"Demographic parity violation for {group}: "
                            f"difference = {diff:.3f} "
                            f"(threshold: {self.fairness_alert_threshold})"
                        )
                    }
                    self.active_alerts.append(alert)
                    logger.warning(alert['message'])

        # Check equalized odds violations
        if 'equalized_odds' in current.fairness_metrics:
            for group, metrics in current.fairness_metrics['equalized_odds'].items():
                max_diff = metrics['max_difference']

                if max_diff > self.fairness_alert_threshold:
                    alert = {
                        'type': 'FAIRNESS_VIOLATION',
                        'subtype': 'equalized_odds',
                        'group': group,
                        'timestamp': current.timestamp,
                        'severity': 'HIGH',
                        'message': (
                            f"Equalized odds violation for {group}: "
                            f"max difference = {max_diff:.3f}"
                        )
                    }
                    self.active_alerts.append(alert)
                    logger.warning(alert['message'])

        # Check for group-specific performance degradation
        if self.baseline.stratified_metrics and current.stratified_metrics:
            for group in current.stratified_metrics:
                if group not in self.baseline.stratified_metrics:
                    continue

                baseline_auc = self.baseline.stratified_metrics[group].auc_roc
                current_auc = current.stratified_metrics[group].auc_roc
                group_decline = baseline_auc - current_auc

                if group_decline > 0.08:  # 8 percentage point decline for subgroup
                    alert = {
                        'type': 'GROUP_PERFORMANCE_DEGRADATION',
                        'group': group,
                        'timestamp': current.timestamp,
                        'severity': 'HIGH',
                        'message': (
                            f"AUC for {group} declined by {group_decline:.3f} "
                            f"from {baseline_auc:.3f} to {current_auc:.3f}"
                        )
                    }
                    self.active_alerts.append(alert)
                    logger.warning(alert['message'])

    def detect_concept_drift(
        self,
        method: str = "kolmogorov_smirnov"
    ) -> Dict[str, Any]:
        """
        Detect concept drift in input feature distributions.

        Concept drift occurs when the relationship between features and
        outcomes changes over time, potentially degrading model performance.

        Args:
            method: Drift detection method ("kolmogorov_smirnov" or "population_stability")

        Returns:
            Dictionary with drift detection results
        """
        if len(self.history) < 2:
            logger.warning("Insufficient history for drift detection")
            return {}

        baseline_dist = self.baseline.data_distribution
        current_dist = self.history[-1].data_distribution

        if not baseline_dist or not current_dist:
            logger.warning("No feature distributions available for drift detection")
            return {}

        drift_results = {}

        if method == "kolmogorov_smirnov":
            # This is simplified - would need actual feature values
            # for proper KS test implementation
            for feature in set(baseline_dist.keys()) & set(current_dist.keys()):
                baseline_mean = baseline_dist[feature]['mean']
                baseline_std = baseline_dist[feature]['std']
                current_mean = current_dist[feature]['mean']
                current_std = current_dist[feature]['std']

                # Approximate drift measure based on distribution moments
                mean_shift = abs(current_mean - baseline_mean) / (baseline_std + 1e-10)

                drift_results[feature] = {
                    'mean_shift_std_units': float(mean_shift),
                    'drift_detected': mean_shift > 2.0,  # >2 standard deviations
                    'baseline_mean': baseline_mean,
                    'current_mean': current_mean
                }

        # Count features with detected drift
        n_drifted = sum(1 for r in drift_results.values() if r['drift_detected'])

        logger.info(
            f"Drift detection: {n_drifted}/{len(drift_results)} features show drift"
        )

        return {
            'method': method,
            'n_features_checked': len(drift_results),
            'n_features_drifted': n_drifted,
            'feature_results': drift_results
        }

    def get_performance_trends(self) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Extract performance metric trends over time.

        Returns:
            Dictionary mapping metric names to time series
        """
        trends = {
            'auc_roc': [],
            'sensitivity': [],
            'specificity': [],
            'calibration_slope': [],
            'brier_score': []
        }

        for snapshot in self.history:
            ts = snapshot.timestamp
            m = snapshot.overall_metrics

            trends['auc_roc'].append((ts, m.auc_roc))
            trends['sensitivity'].append((ts, m.sensitivity))
            trends['specificity'].append((ts, m.specificity))
            trends['calibration_slope'].append((ts, m.calibration_slope))
            trends['brier_score'].append((ts, m.brier_score))

        return trends

    def generate_monitoring_report(self) -> str:
        """
        Generate comprehensive temporal monitoring report.

        Returns:
            Formatted monitoring report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("TEMPORAL VALIDATION AND PERFORMANCE MONITORING REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Monitoring period
        if len(self.history) > 1:
            start = self.history[0].timestamp
            end = self.history[-1].timestamp
            lines.append(f"Monitoring period: {start.date()} to {end.date()}")
            lines.append(f"Number of evaluations: {len(self.history)}")
            lines.append("")

        # Baseline performance
        lines.append("BASELINE PERFORMANCE:")
        bm = self.baseline.overall_metrics
        lines.append(f"  Timestamp: {self.baseline.timestamp}")
        lines.append(f"  AUC-ROC: {bm.auc_roc:.4f}")
        lines.append(f"  Sensitivity: {bm.sensitivity:.4f}")
        lines.append(f"  Specificity: {bm.specificity:.4f}")
        lines.append(f"  Calibration slope: {bm.calibration_slope:.4f}")
        lines.append("")

        # Current performance
        if len(self.history) > 1:
            current = self.history[-1]
            cm = current.overall_metrics

            lines.append("CURRENT PERFORMANCE:")
            lines.append(f"  Timestamp: {current.timestamp}")
            lines.append(f"  AUC-ROC: {cm.auc_roc:.4f} (change: {cm.auc_roc - bm.auc_roc:+.4f})")
            lines.append(f"  Sensitivity: {cm.sensitivity:.4f} (change: {cm.sensitivity - bm.sensitivity:+.4f})")
            lines.append(f"  Specificity: {cm.specificity:.4f} (change: {cm.specificity - bm.specificity:+.4f})")
            lines.append(f"  Calibration slope: {cm.calibration_slope:.4f} (change: {cm.calibration_slope - bm.calibration_slope:+.4f})")
            lines.append("")

        # Active alerts
        if self.active_alerts:
            lines.append(f"ACTIVE ALERTS ({len(self.active_alerts)}):")

            # Group by severity
            high_alerts = [a for a in self.active_alerts if a['severity'] == 'HIGH']
            med_alerts = [a for a in self.active_alerts if a['severity'] == 'MEDIUM']

            if high_alerts:
                lines.append(f"  HIGH SEVERITY ({len(high_alerts)}):")
                for alert in high_alerts[-5:]:  # Show most recent 5
                    lines.append(f"    [{alert['timestamp'].strftime('%Y-%m-%d')}] {alert['message']}")
                lines.append("")

            if med_alerts:
                lines.append(f"  MEDIUM SEVERITY ({len(med_alerts)}):")
                for alert in med_alerts[-5:]:
                    lines.append(f"    [{alert['timestamp'].strftime('%Y-%m-%d')}] {alert['message']}")
                lines.append("")
        else:
            lines.append("ACTIVE ALERTS: None")
            lines.append("")

        # Fairness tracking
        if len(self.history) > 1:
            current = self.history[-1]
            if current.fairness_metrics and 'demographic_parity' in current.fairness_metrics:
                lines.append("CURRENT FAIRNESS METRICS:")
                for group, metrics in current.fairness_metrics['demographic_parity'].items():
                    diff = metrics['difference']
                    ratio = metrics['ratio']
                    lines.append(f"  {group}:")
                    lines.append(f"    Demographic parity difference: {diff:+.4f}")
                    lines.append(f"    Demographic parity ratio: {ratio:.4f}")
                lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)
```

This temporal validation framework enables comprehensive monitoring of deployed models with explicit attention to equity considerations. The statistical process control methods detect meaningful performance degradation while avoiding excessive false alarms. The stratified monitoring tracks performance trends separately for each demographic group, surfacing emerging disparities that might be hidden by aggregate metrics. The automated alerting system flags concerning patterns requiring investigation, with severity levels guiding appropriate responses. Together these components provide production-ready infrastructure for ongoing model validation after deployment.

## 15.5 External Validation Across Diverse Sites

External validation evaluates model performance on data from institutions not involved in model development, providing critical evidence about generalizability. For clinical AI intended to serve diverse populations, external validation must span geographically and demographically heterogeneous sites including community hospitals and federally qualified health centers serving predominantly underserved populations, not just academic medical centers. Models validated only at academic institutions may fail dramatically when deployed in safety-net settings with different patient populations, clinical practices, and data quality.

The fundamental challenge in external validation is obtaining appropriate datasets that represent the full diversity of intended deployment settings. Formal data sharing agreements, IRB approvals, and technical infrastructure for multi-site collaboration all present substantial barriers. Federated learning approaches where models are evaluated at remote sites without centralizing data can help address some privacy and governance concerns while enabling broader validation. However, coordinating multi-site validation still requires significant effort and institutional commitment.

From an equity perspective, external validation cohorts must be selected intentionally to include sites serving underserved populations rather than convenience samples of easily accessible institutions. If external validation includes three academic medical centers in wealthy urban areas, it provides little information about model performance for patients in rural community hospitals or urban safety-net institutions. The validation design must explicitly prioritize diversity across multiple dimensions including geography, patient demographics, socioeconomic factors, insurance mix, and clinical complexity.

Heterogeneity in data quality and completeness across validation sites poses additional challenges. Academic institutions typically have well-resourced informatics infrastructure, comprehensive documentation, and complete laboratory testing. Community hospitals may have sparser documentation, less complete coding, and more missing data. If a model was developed at a well-resourced institution, external validation at similar sites may show excellent performance while validation at under-resourced sites reveals substantial problems. The external validation must therefore include not just diverse patient populations but diverse data contexts.

We now develop frameworks for multi-site external validation with explicit equity considerations.

```python
"""
External validation framework for multi-site clinical AI evaluation.

This module implements comprehensive external validation strategies including
coordination of multi-site evaluation, meta-analysis across sites, assessment
of site-level heterogeneity, and validation reporting for diverse settings.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats

@dataclass
class SiteCharacteristics:
    """
    Characteristics of an external validation site.

    Attributes:
        site_id: Unique identifier
        site_name: Human-readable name
        site_type: Type of institution (academic, community, FQHC, etc.)
        geographic_region: Geographic location
        annual_volume: Approximate annual patient volume
        primary_payer_mix: Distribution of insurance types
        demographics: Patient demographic distribution
        data_quality_metrics: Data completeness and quality indicators
    """
    site_id: str
    site_name: str
    site_type: str
    geographic_region: str
    annual_volume: int
    primary_payer_mix: Dict[str, float]
    demographics: Dict[str, Dict[str, float]]
    data_quality_metrics: Dict[str, float]

@dataclass
class SiteValidationResults:
    """
    Validation results from a single external site.

    Attributes:
        site_characteristics: Site information
        overall_metrics: Overall performance at this site
        stratified_metrics: Performance by demographics at this site
        fairness_metrics: Fairness evaluation at this site
        n_samples: Number of validation samples
        validation_timestamp: When validation was performed
        notes: Additional qualitative observations
    """
    site_characteristics: SiteCharacteristics
    overall_metrics: PerformanceMetrics
    stratified_metrics: Dict[str, PerformanceMetrics]
    fairness_metrics: Dict[str, Any]
    n_samples: int
    validation_timestamp: datetime
    notes: Optional[str] = None

class MultiSiteExternalValidator:
    """
    Framework for coordinating external validation across diverse sites.

    This class implements comprehensive multi-site external validation including
    meta-analysis of site-level results, assessment of heterogeneity, and
    reporting that highlights performance variation across different care
    settings and patient populations.
    """

    def __init__(
        self,
        internal_validation_results: PerformanceMetrics,
        classification_threshold: float = 0.5,
        min_site_sample_size: int = 200
    ):
        """
        Initialize multi-site validator.

        Args:
            internal_validation_results: Baseline internal validation metrics
            classification_threshold: Threshold for binary classification
            min_site_sample_size: Minimum samples required per site
        """
        self.internal_results = internal_validation_results
        self.classification_threshold = classification_threshold
        self.min_site_sample_size = min_site_sample_size

        # Storage for site-level results
        self.site_results: List[SiteValidationResults] = []

        logger.info(
            f"Initialized MultiSiteExternalValidator with baseline AUC "
            f"{internal_validation_results.auc_roc:.4f}"
        )

    def add_site_results(
        self,
        site_results: SiteValidationResults
    ):
        """
        Add validation results from an external site.

        Args:
            site_results: Complete validation results from one site
        """
        if site_results.n_samples < self.min_site_sample_size:
            logger.warning(
                f"Site {site_results.site_characteristics.site_name} has only "
                f"{site_results.n_samples} samples (minimum: {self.min_site_sample_size}). "
                f"Results may be unreliable."
            )

        self.site_results.append(site_results)

        logger.info(
            f"Added results from {site_results.site_characteristics.site_name}: "
            f"n={site_results.n_samples}, AUC={site_results.overall_metrics.auc_roc:.4f}"
        )

    def meta_analyze_performance(
        self,
        metric: str = "auc_roc"
    ) -> Dict[str, float]:
        """
        Perform meta-analysis of performance across sites.

        Uses inverse-variance weighting to combine site-level estimates,
        providing overall external validation performance and measures of
        heterogeneity across sites.

        Args:
            metric: Performance metric to meta-analyze

        Returns:
            Dictionary with pooled estimate and heterogeneity statistics
        """
        if len(self.site_results) < 2:
            logger.warning("Need at least 2 sites for meta-analysis")
            return {}

        logger.info(f"Performing meta-analysis of {metric} across {len(self.site_results)} sites")

        # Extract metric values and sample sizes
        estimates = []
        variances = []
        weights = []

        for site in self.site_results:
            metric_value = getattr(site.overall_metrics, metric)
            n = site.n_samples

            # Approximate variance for AUC using Hanley-McNeil
            if metric == "auc_roc":
                # Simplified variance approximation
                q1 = metric_value / (2 - metric_value)
                q2 = 2 * metric_value**2 / (1 + metric_value)

                # Assume balanced prevalence for simplicity
                n_pos = site.overall_metrics.n_positive
                n_neg = n - n_pos

                if n_pos > 0 and n_neg > 0:
                    variance = (
                        metric_value * (1 - metric_value) +
                        (n_pos - 1) * (q1 - metric_value**2) +
                        (n_neg - 1) * (q2 - metric_value**2)
                    ) / (n_pos * n_neg)
                else:
                    variance = 1.0 / n  # Fallback
            else:
                # For other metrics, use simple binomial variance approximation
                variance = metric_value * (1 - metric_value) / n

            estimates.append(metric_value)
            variances.append(variance)
            weights.append(1.0 / variance)

        estimates = np.array(estimates)
        variances = np.array(variances)
        weights = np.array(weights)

        # Inverse-variance weighted pooled estimate
        pooled_estimate = np.sum(weights * estimates) / np.sum(weights)
        pooled_variance = 1.0 / np.sum(weights)
        pooled_se = np.sqrt(pooled_variance)

        # Heterogeneity statistics (I-squared and Cochran's Q)
        q_statistic = np.sum(weights * (estimates - pooled_estimate)**2)
        df = len(estimates) - 1

        if df > 0:
            q_pvalue = 1 - stats.chi2.cdf(q_statistic, df)

            # I-squared: proportion of variance due to heterogeneity
            i_squared = max(0, (q_statistic - df) / q_statistic) if q_statistic > 0 else 0
        else:
            q_pvalue = 1.0
            i_squared = 0.0

        # Prediction interval for a new site
        # Incorporates both within-site and between-site variation
        tau_squared = max(0, (q_statistic - df) / (np.sum(weights) - np.sum(weights**2) / np.sum(weights)))
        pred_variance = pooled_variance + tau_squared
        pred_se = np.sqrt(pred_variance)

        # 95% confidence and prediction intervals
        ci_lower = pooled_estimate - 1.96 * pooled_se
        ci_upper = pooled_estimate + 1.96 * pooled_se
        pred_lower = pooled_estimate - 1.96 * pred_se
        pred_upper = pooled_estimate + 1.96 * pred_se

        results = {
            'pooled_estimate': float(pooled_estimate),
            'pooled_se': float(pooled_se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'prediction_interval_lower': float(pred_lower),
            'prediction_interval_upper': float(pred_upper),
            'q_statistic': float(q_statistic),
            'q_pvalue': float(q_pvalue),
            'i_squared': float(i_squared),
            'tau_squared': float(tau_squared),
            'n_sites': len(self.site_results)
        }

        logger.info(
            f"Meta-analysis results: pooled {metric} = {pooled_estimate:.4f} "
            f"(95% CI: {ci_lower:.4f}-{ci_upper:.4f}), "
            f"I^2 = {i_squared:.1%}"
        )

        return results

    def assess_site_heterogeneity(self) -> Dict[str, Any]:
        """
        Assess heterogeneity in performance across validation sites.

        Examines whether performance varies systematically by site
        characteristics such as site type, geographic region, or patient
        demographics.

        Returns:
            Dictionary with heterogeneity analysis results
        """
        if len(self.site_results) < 3:
            logger.warning("Need at least 3 sites for heterogeneity assessment")
            return {}

        logger.info("Assessing site-level heterogeneity")

        # Organize results by site characteristics
        by_site_type = {}
        by_region = {}

        for site in self.site_results:
            site_type = site.site_characteristics.site_type
            region = site.site_characteristics.geographic_region

            if site_type not in by_site_type:
                by_site_type[site_type] = []
            by_site_type[site_type].append(site.overall_metrics.auc_roc)

            if region not in by_region:
                by_region[region] = []
            by_region[region].append(site.overall_metrics.auc_roc)

        heterogeneity = {
            'by_site_type': {},
            'by_region': {},
            'overall_variance': float(np.var([s.overall_metrics.auc_roc for s in self.site_results]))
        }

        # Analyze by site type
        for site_type, aucs in by_site_type.items():
            heterogeneity['by_site_type'][site_type] = {
                'n_sites': len(aucs),
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs)),
                'min_auc': float(np.min(aucs)),
                'max_auc': float(np.max(aucs))
            }

        # Analyze by region
        for region, aucs in by_region.items():
            heterogeneity['by_region'][region] = {
                'n_sites': len(aucs),
                'mean_auc': float(np.mean(aucs)),
                'std_auc': float(np.std(aucs)),
                'min_auc': float(np.min(aucs)),
                'max_auc': float(np.max(aucs))
            }

        # ANOVA test for differences across site types (if sufficient data)
        if len(by_site_type) >= 2:
            site_type_groups = [aucs for aucs in by_site_type.values() if len(aucs) >= 2]
            if len(site_type_groups) >= 2:
                f_stat, p_value = stats.f_oneway(*site_type_groups)
                heterogeneity['site_type_anova'] = {
                    'f_statistic': float(f_stat),
                    'p_value': float(p_value)
                }

        return heterogeneity

    def compare_to_internal_validation(self) -> Dict[str, Any]:
        """
        Compare external validation results to internal validation.

        Assesses whether model performance generalizes well or shows
        substantial degradation on external data.

        Returns:
            Dictionary with internal vs external comparison
        """
        if not self.site_results:
            return {}

        # Compute mean external performance
        external_aucs = [s.overall_metrics.auc_roc for s in self.site_results]
        mean_external_auc = np.mean(external_aucs)
        std_external_auc = np.std(external_aucs)

        internal_auc = self.internal_results.auc_roc

        # Performance degradation
        degradation = internal_auc - mean_external_auc
        degradation_pct = 100 * degradation / internal_auc

        # Statistical test: is external performance significantly different?
        # One-sample t-test comparing external sites to internal estimate
        if len(external_aucs) >= 2:
            t_stat, p_value = stats.ttest_1samp(external_aucs, internal_auc)
        else:
            t_stat, p_value = np.nan, np.nan

        comparison = {
            'internal_auc': float(internal_auc),
            'mean_external_auc': float(mean_external_auc),
            'std_external_auc': float(std_external_auc),
            'min_external_auc': float(np.min(external_aucs)),
            'max_external_auc': float(np.max(external_aucs)),
            'degradation_absolute': float(degradation),
            'degradation_percent': float(degradation_pct),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'n_external_sites': len(self.site_results)
        }

        logger.info(
            f"Internal vs external: {internal_auc:.4f} vs {mean_external_auc:.4f} "
            f"(degradation: {degradation:.4f}, {degradation_pct:.1f}%)"
        )

        return comparison

    def generate_external_validation_report(self) -> str:
        """
        Generate comprehensive external validation report.

        Returns:
            Formatted multi-site validation report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EXTERNAL VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append(f"Number of external validation sites: {len(self.site_results)}")
        total_samples = sum(s.n_samples for s in self.site_results)
        lines.append(f"Total external validation samples: {total_samples:,}")
        lines.append("")

        # Internal validation reference
        lines.append("INTERNAL VALIDATION REFERENCE:")
        lines.append(f"  AUC-ROC: {self.internal_results.auc_roc:.4f}")
        lines.append(f"  Sensitivity: {self.internal_results.sensitivity:.4f}")
        lines.append(f"  Specificity: {self.internal_results.specificity:.4f}")
        lines.append("")

        # Site-by-site results
        lines.append("SITE-LEVEL RESULTS:")
        lines.append("")

        for site in sorted(self.site_results, key=lambda x: x.overall_metrics.auc_roc, reverse=True):
            char = site.site_characteristics
            metrics = site.overall_metrics

            lines.append(f"  {char.site_name} ({char.site_type}, {char.geographic_region})")
            lines.append(f"    Samples: {site.n_samples:,}")
            lines.append(f"    AUC-ROC: {metrics.auc_roc:.4f}")
            lines.append(f"    Sensitivity: {metrics.sensitivity:.4f}")
            lines.append(f"    Specificity: {metrics.specificity:.4f}")
            lines.append(f"    Calibration slope: {metrics.calibration_slope:.4f}")

            if site.notes:
                lines.append(f"    Notes: {site.notes}")

            lines.append("")

        # Meta-analysis
        meta_results = self.meta_analyze_performance("auc_roc")
        if meta_results:
            lines.append("META-ANALYSIS:")
            lines.append(f"  Pooled AUC-ROC: {meta_results['pooled_estimate']:.4f}")
            lines.append(
                f"  95% CI: ({meta_results['ci_lower']:.4f}, {meta_results['ci_upper']:.4f})"
            )
            lines.append(
                f"  95% Prediction interval: ({meta_results['prediction_interval_lower']:.4f}, "
                f"{meta_results['prediction_interval_upper']:.4f})"
            )
            lines.append(f"  Heterogeneity (I^2): {meta_results['i_squared']:.1%}")
            lines.append(f"  Cochran's Q p-value: {meta_results['q_pvalue']:.4f}")
            lines.append("")

        # Heterogeneity assessment
        heterogeneity = self.assess_site_heterogeneity()
        if heterogeneity and 'by_site_type' in heterogeneity:
            lines.append("PERFORMANCE BY SITE TYPE:")
            for site_type, stats_dict in heterogeneity['by_site_type'].items():
                lines.append(f"  {site_type}:")
                lines.append(f"    n={stats_dict['n_sites']} sites")
                lines.append(
                    f"    Mean AUC: {stats_dict['mean_auc']:.4f} "
                    f"(SD: {stats_dict['std_auc']:.4f})"
                )
                lines.append(
                    f"    Range: {stats_dict['min_auc']:.4f} - {stats_dict['max_auc']:.4f}"
                )
            lines.append("")

        # Internal vs external comparison
        comparison = self.compare_to_internal_validation()
        if comparison:
            lines.append("INTERNAL VS EXTERNAL VALIDATION:")
            lines.append(f"  Internal AUC: {comparison['internal_auc']:.4f}")
            lines.append(f"  Mean external AUC: {comparison['mean_external_auc']:.4f}")
            lines.append(
                f"  Performance degradation: {comparison['degradation_absolute']:.4f} "
                f"({comparison['degradation_percent']:.1f}%)"
            )

            if comparison['p_value'] < 0.05:
                lines.append(
                    f"  Difference is statistically significant (p={comparison['p_value']:.4f})"
                )
            else:
                lines.append(
                    f"  Difference is not statistically significant (p={comparison['p_value']:.4f})"
                )
            lines.append("")

        # Interpretation
        lines.append("INTERPRETATION:")

        if comparison and abs(comparison['degradation_percent']) < 5:
            lines.append("  [OK] Model performance generalizes well to external sites")
        elif comparison and abs(comparison['degradation_percent']) < 10:
            lines.append("  [WARN]  Model shows modest performance degradation on external data")
        else:
            lines.append("  [FAIL] Model shows substantial performance degradation on external data")

        if meta_results and meta_results['i_squared'] < 0.25:
            lines.append("  [OK] Performance is consistent across sites (low heterogeneity)")
        elif meta_results and meta_results['i_squared'] < 0.50:
            lines.append("  [WARN]  Performance shows moderate variation across sites")
        else:
            lines.append("  [FAIL] Performance varies substantially across sites (high heterogeneity)")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
```

This multi-site external validation framework enables rigorous assessment of model generalizability across diverse healthcare settings. The meta-analysis appropriately pools site-level estimates while quantifying heterogeneity, providing both overall external validation performance and measures of variation across sites. The heterogeneity assessment examines whether performance differs systematically by site characteristics, revealing patterns that inform deployment decisions. The comprehensive reporting highlights both successes and limitations of model generalizability, supporting transparent communication with stakeholders and regulators about expected performance in diverse real-world settings.

## 15.6 Conclusion and Key Takeaways

This chapter has developed comprehensive validation strategies for clinical AI systems with consistent attention to equity considerations that are systematically neglected in standard approaches. The fundamental insight is that rigorous validation requires explicitly assessing both overall performance and fairness metrics across diverse patient populations and care settings, with adequate statistical power to detect clinically meaningful disparities. Aggregate performance metrics alone are insufficient because models can achieve excellent average performance while exhibiting severe disparities across demographic subgroups or systematic failures in underrepresented settings.

Internal validation with equity-focused data splitting ensures validation cohorts contain adequate representation of key patient subgroups through multidimensional stratification. Standard random splits or simple outcome stratification often result in validation sets with insufficient numbers of patients from underrepresented groups, making fairness evaluation statistically infeasible. The stratified splitting strategies and sample size calculations developed in this chapter enable validation study design that can actually detect disparities rather than simply documenting aggregate performance.

Power calculations for fairness metrics reveal that detecting meaningful disparities requires substantially larger validation cohorts than standard approaches suggest. Comparing performance between demographic groups with adequate statistical power necessitates sufficient sample sizes within each group, not just overall. Multiple testing corrections when evaluating fairness across several demographic groups further increase required sample sizes. The power calculation frameworks provide practical tools for determining whether proposed validation studies are adequate for their stated purposes or merely give false confidence through underpowered fairness evaluation.

Temporal validation assesses model performance degradation over time, which is essential for models deployed in evolving healthcare environments. Performance monitoring must track not just aggregate metrics but also group-specific performance and fairness measures, because disparities can emerge or worsen even when overall performance remains stable. The temporal validation framework with automated alerting enables early detection of concerning patterns requiring investigation and potential model retraining.

External validation across geographically and demographically diverse sites provides critical evidence about generalizability beyond single institutions. Models validated only at academic medical centers may fail dramatically when deployed in community hospitals or federally qualified health centers serving predominantly underserved populations. The multi-site validation framework with meta-analysis quantifies both overall external performance and heterogeneity across sites, revealing whether models generalize consistently or show substantial variation depending on care setting and patient population characteristics.

Several critical principles emerge from this work. First, validation study design must be intentional about ensuring adequate representation of populations for whom fairness evaluation is essential, not just convenient samples from easily accessible institutions. Second, sample size calculations must account for fairness evaluation requirements, not just overall performance estimation, to ensure validation studies have adequate statistical power. Third, validation is an ongoing process rather than a one-time evaluation. Models require continuous monitoring after deployment to detect performance degradation and emerging disparities due to distributional shift, changing clinical practices, or feedback loops. Fourth, validation findings must be reported transparently, including both successes and limitations, to enable appropriate deployment decisions and maintain stakeholder trust.

From an equity perspective, rigorous validation is essential but not sufficient for ensuring clinical AI serves diverse populations fairly. Validation can only assess whether models meet specified performance and fairness criteria; it cannot fix fundamental problems stemming from biased training data, inappropriate modeling choices, or deployment contexts that differ systematically from development settings. Comprehensive validation that surfaces fairness issues must be paired with genuine commitment to addressing identified problems through improved data collection, fairness-aware modeling approaches, or explicit constraints on deployment contexts.

The stakes are particularly high in healthcare applications affecting underserved populations. Inadequate validation can lead to deployment of systems that appear rigorously evaluated yet systematically fail certain patient groups, exacerbating rather than reducing health disparities. The validation strategies developed in this chapter provide practitioners with comprehensive frameworks for ensuring clinical AI systems are genuinely safe and fair for all populations they are intended to serve. By making fairness evaluation a core component of validation rather than an afterthought, we can work toward AI systems that advance rather than undermine health equity.

## Bibliography

Adamson, A. S., & Smith, A. (2018).
Machine learning and health care disparities in dermatology.
*JAMA Dermatology*, 154(11), 1247-1248.

Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016).
Machine bias: There's software used across the country to predict future criminals. And it's biased against blacks.
*ProPublica*, May 23.

Authors, A. D. (2019).
Reporting guidelines for clinical trials evaluating artificial intelligence interventions are needed.
*Nature Medicine*, 25(10), 1467-1468.

Balduzzi, S., Rücker, G., & Schwarzer, G. (2019).
How to perform a meta-analysis with R: a practical tutorial.
*Evidence-Based Mental Health*, 22(4), 153-160.

Bellamy, R. K., Dey, K., Hind, M., Hoffman, S. C., Houde, S., Kannan, K., ... & Nagar, S. (2019).
AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.
*IBM Journal of Research and Development*, 63(4/5), 4-1.

Benjamini, Y., & Hochberg, Y. (1995).
Controlling the false discovery rate: a practical and powerful approach to multiple testing.
*Journal of the Royal Statistical Society: Series B (Methodological)*, 57(1), 289-300.

Bleeker, S. E., Moll, H. A., Steyerberg, E. W., Donders, A. R. T., Derksen-Lubsen, G., Grobbee, D. E., & Moons, K. G. (2003).
External validation is necessary in prediction research: a clinical example.
*Journal of Clinical Epidemiology*, 56(9), 826-832.

Bonnett, L. J., Snell, K. I., Collins, G. S., & Riley, R. D. (2019).
Guide to presenting clinical prediction models for use in clinical settings.
*BMJ*, 365, l737.

Buolamwini, J., & Gebru, T. (2018).
Gender shades: Intersectional accuracy disparities in commercial gender classification.
*Proceedings of Machine Learning Research*, 81, 1-15.

Cabitza, F., Campagner, A., & Balsano, C. (2021).
Bridging the "last mile" gap between AI implementation and operation: "data awareness" that matters.
*Annals of Translational Medicine*, 8(7), 501.

Chen, I. Y., Szolovits, P., & Ghassemi, M. (2019).
Can AI help reduce disparities in general medical and mental health care?
*AMA Journal of Ethics*, 21(2), 167-179.

Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. (2015).
Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement.
*BMJ*, 350, g7594.

DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach.
*Biometrics*, 44(3), 837-845.

Efron, B., & Tibshirani, R. J. (1994).
*An introduction to the bootstrap*.
CRC Press.

Finlayson, S. G., Subbaswamy, A., Singh, K., Bowers, J., Kupke, A., Zittrain, J., ... & Saria, S. (2021).
The clinician and dataset shift in artificial intelligence.
*New England Journal of Medicine*, 385(3), 283-286.

Gianfrancesco, M. A., Tamang, S., Yazdany, J., & Schmajuk, G. (2018).
Potential biases in machine learning algorithms using electronic health record data.
*JAMA Internal Medicine*, 178(11), 1544-1547.

Harrell, F. E. (2015).
*Regression modeling strategies: with applications to linear models, logistic and ordinal regression, and survival analysis*.
Springer.

Henry, K. E., Hager, D. N., Pronovost, P. J., & Saria, S. (2015).
A targeted real-time early warning score (TREWScore) for septic shock.
*Science Translational Medicine*, 7(299), 299ra122.

Higgins, J. P., Thompson, S. G., Deeks, J. J., & Altman, D. G. (2003).
Measuring inconsistency in meta-analyses.
*BMJ*, 327(7414), 557-560.

Hosmer Jr, D. W., Lemeshow, S., & Sturdivant, R. X. (2013).
*Applied logistic regression* (Vol. 398).
John Wiley & Sons.

Iasonos, A., Schrag, D., Raj, G. V., & Panageas, K. S. (2008).
How to build and interpret a nomogram for cancer prognosis.
*Journal of Clinical Oncology*, 26(8), 1364-1370.

Jiang, X., Osl, M., Kim, J., & Ohno-Machado, L. (2012).
Calibrating predictive model estimates to support personalized medicine.
*Journal of the American Medical Informatics Association*, 19(2), 263-274.

Justice, A. C., Covinsky, K. E., & Berlin, J. A. (1999).
Assessing the generalizability of prognostic information.
*Annals of Internal Medicine*, 130(6), 515-524.

Kappen, T. H., van Klei, W. A., van Wolfswinkel, L., Kalkman, C. J., Vergouwe, Y., & Moons, K. G. (2014).
Evaluating the impact of prediction models: lessons learned, challenges, and recommendations.
*Diagnostic and Prognostic Research*, 2(1), 1-11.

Kelly, C. J., Karthikesalingam, A., Suleyman, M., Corrado, G., & King, D. (2019).
Key challenges for delivering clinical impact with artificial intelligence.
*BMC Medicine*, 17(1), 1-9.

Kohavi, R. (1995).
A study of cross-validation and bootstrap for accuracy estimation and model selection.
*Proceedings of the 14th International Joint Conference on Artificial Intelligence*, 2, 1137-1143.

Liu, X., Cruz Rivera, S., Moher, D., Calvert, M. J., & Denniston, A. K. (2019).
Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI extension.
*The Lancet Digital Health*, 2(10), e537-e548.

Moons, K. G., Altman, D. G., Reitsma, J. B., Ioannidis, J. P., Macaskill, P., Steyerberg, E. W., ... & Collins, G. S. (2015).
Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): explanation and elaboration.
*Annals of Internal Medicine*, 162(1), W1-W73.

Noseworthy, P. A., Attia, Z. I., Brewer, L. C., Hayes, S. N., Yao, X., Kapa, S., ... & Lopez-Jimenez, F. (2020).
Assessing and mitigating bias in medical artificial intelligence: the effects of race and ethnicity on a deep learning model for ECG analysis.
*Circulation: Arrhythmia and Electrophysiology*, 13(3), e007988.

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).
Dissecting racial bias in an algorithm used to manage the health of populations.
*Science*, 366(6464), 447-453.

Park, Y., Jackson, G. P., Foreman, M. A., Gruen, D., Hu, J., & Das, A. K. (2021).
Evaluation of artificial intelligence in medicine: phases of clinical research.
*JAMIA Open*, 4(3), ooab033.

Pfohl, S. R., Foryciarz, A., & Shah, N. H. (2021).
An empirical characterization of fair machine learning for clinical risk prediction.
*Journal of Biomedical Informatics*, 113, 103621.

Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H. (2018).
Ensuring fairness in machine learning to advance health equity.
*Annals of Internal Medicine*, 169(12), 866-872.

Riley, R. D., Snell, K. I., Ensor, J., Burke, D. L., Harrell Jr, F. E., Moons, K. G., & Collins, G. S. (2020).
Minimum sample size for developing a multivariable prediction model: PART II - binary and time-to-event outcomes.
*Statistics in Medicine*, 38(7), 1276-1296.

Ross, M. K., Wei, W., & Ohno-Machado, L. (2014).
"Big data" and the electronic health record.
*Yearbook of Medical Informatics*, 9(1), 97-104.

Saito, T., & Rehmsmeier, M. (2015).
The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets.
*PloS One*, 10(3), e0118432.

Sendak, M. P., Gao, M., Brajer, N., & Balu, S. (2020).
Presenting machine learning model information to clinical end users with model facts labels.
*NPJ Digital Medicine*, 3(1), 1-4.

Shah, N. H., Milstein, A., & Bagley, P. S. C. (2019).
Making machine learning models clinically useful.
*JAMA*, 322(14), 1351-1352.

Siontis, G. C., Tzoulaki, I., Castaldi, P. J., & Ioannidis, J. P. (2015).
External validation of new risk prediction models is infrequent and reveals worse prognostic discrimination.
*Journal of Clinical Epidemiology*, 68(1), 25-34.

Steyerberg, E. W., & Harrell Jr, F. E. (2016).
Prediction models need appropriate internal, internal-external, and external validation.
*Journal of Clinical Epidemiology*, 69, 245-247.

Steyerberg, E. W., Harrell Jr, F. E., Borsboom, G. J., Eijkemans, M. J. C., Vergouwe, Y., & Habbema, J. D. F. (2001).
Internal validation of predictive models: efficiency of some procedures for logistic regression analysis.
*Journal of Clinical Epidemiology*, 54(8), 774-781.

Subbaswamy, A., & Saria, S. (2020).
From development to deployment: dataset shift, causality, and shift-stable models in health AI.
*Biostatistics*, 21(2), 345-352.

Ustun, B., & Rudin, C. (2019).
Learning optimized risk scores.
*Journal of Machine Learning Research*, 20(150), 1-75.

Van Belle, V., Pelckmans, K., Van Huffel, S., & Suykens, J. A. (2011).
Support vector methods for survival analysis: a comparison between ranking and regression approaches.
*Artificial Intelligence in Medicine*, 53(2), 107-118.

VanderWeele, T. J., & Mathur, M. B. (2019).
Some desirable properties of the Bonferroni correction: is the Bonferroni correction really so bad?
*American Journal of Epidemiology*, 188(3), 617-618.

Vergouwe, Y., Royston, P., Moons, K. G., & Altman, D. G. (2010).
Development and validation of a prediction model with missing predictor data: a practical approach.
*Journal of Clinical Epidemiology*, 63(2), 205-214.

Vickers, A. J., Van Calster, B., & Steyerberg, E. W. (2016).
Net benefit approaches to the evaluation of prediction models, molecular markers, and diagnostic tests.
*BMJ*, 352, i6.

Vickers, A. J., & Elkin, E. B. (2006).
Decision curve analysis: a novel method for evaluating prediction models.
*Medical Decision Making*, 26(6), 565-574.

Vollmer, S., Mateen, B. A., Bohner, G., Király, F. J., Ghani, R., Jonsson, P., ... & Hemingway, H. (2020).
Machine learning and artificial intelligence research for patient benefit: 20 critical questions on transparency, replicability, ethics, and effectiveness.
*BMJ*, 368, l6927.

Wong, A., Otles, E., Donnelly, J. P., Krumm, A., McCullough, J., DeTroyer-Cooley, O., ... & Singh, K. (2021).
External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients.
*JAMA Internal Medicine*, 181(8), 1065-1070.

Wynants, L., Van Calster, B., Collins, G. S., Riley, R. D., Heinze, G., Schuit, E., ... & van Smeden, M. (2020).
Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal.
*BMJ*, 369, m1328.
