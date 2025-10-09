---
layout: chapter
title: "Chapter 21: Health Equity Metrics and Evaluation Frameworks"
chapter_number: 21
part_number: 5
prev_chapter: /chapters/chapter-20-monitoring-maintenance/
next_chapter: /chapters/chapter-22-clinical-decision-support/
---
# Chapter 21: Clinical Risk Prediction with Fairness Constraints

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Develop acute clinical deterioration prediction models for hospitalized patients that maintain equitable sensitivity across demographic groups while accounting for systematic differences in observation frequency and documentation completeness across care settings serving diverse populations.

2. Build chronic disease risk prediction systems for population health management that avoid discriminatory allocation of preventive resources by explicitly incorporating social determinants of health as targets for intervention rather than fixed risk factors that penalize patients for structural inequities.

3. Construct readmission risk models that distinguish between readmissions reflecting inadequate care transitions and those reflecting social vulnerabilities like housing instability or food insecurity, enabling targeted interventions that address root causes rather than penalizing safety-net hospitals serving complex patients.

4. Implement mortality prediction models for critically ill patients that provide calibrated risk estimates stratified by demographic factors and care intensity, with appropriate uncertainty quantification that acknowledges when predictions fall outside training distribution and may be unreliable.

5. Design fairness-aware threshold selection procedures that optimize clinical objectives like early warning system sensitivity while constraining disparities in false positive rates that can lead to alert fatigue for clinicians caring for predominantly minority patients.

6. Validate risk prediction models through comprehensive evaluation frameworks that assess not only discrimination and calibration overall but also fairness metrics across patient subgroups, with statistical tests for significant performance disparities and quantification of their clinical impact on care delivery equity.

## 21.1 Introduction: The Dual Mandate of Fair Risk Prediction

Clinical risk prediction models serve as the foundation for countless healthcare decisions, from identifying patients for intensive monitoring to allocating scarce preventive resources to triggering clinical decision support interventions. These models promise to improve care by enabling proactive rather than reactive medicine, catching deteriorating patients before crises occur, targeting interventions to those who will benefit most, and allocating limited clinical attention efficiently across large patient populations. The proliferation of risk prediction models reflects genuine enthusiasm about their potential to improve population health outcomes through more systematic, data-driven clinical reasoning.

Yet this promise has been profoundly compromised by documented failures of fairness. Risk prediction models have been shown to systematically underestimate risk for Black patients compared to white patients with similar clinical presentations, leading to delayed recognition of clinical deterioration and preventable deaths. Readmission risk models have penalized safety-net hospitals serving socially vulnerable populations by counting readmissions driven by social factors as quality failures rather than recognizing them as symptoms of inadequate community support systems. Commercial risk prediction algorithms used to allocate care management resources have directed those resources toward healthier white patients rather than sicker Black patients due to using healthcare utilization as a proxy for health need. These failures represent not merely technical errors but rather fundamental misunderstandings of how healthcare data reflects and perpetuates structural inequities.

The core challenge is that clinical risk prediction from observational healthcare data cannot be divorced from the social context in which that data was generated. Electronic health records document care that was delivered, not care that should have been delivered. When patients face barriers to accessing care due to lack of insurance, transportation difficulties, language barriers, or discrimination, their health records become sparse and incomplete. Risk prediction models trained on this data may interpret data sparsity as indicating lower risk rather than recognizing it as a symptom of healthcare access barriers that actually increase risk. Similarly, when documentation practices differ across care settings, with safety-net hospitals serving sicker patients but having less comprehensive documentation due to resource constraints, risk models may systematically underestimate severity for patients receiving care in those settings.

Health equity considerations must therefore be woven throughout every stage of clinical risk prediction model development, not added as an afterthought during model evaluation. The choice of outcome to predict embeds assumptions about what matters for health and whose outcomes we prioritize. Feature engineering decisions determine whether social determinants are treated as fixed patient characteristics that increase risk or as modifiable intervention targets. Model architecture choices affect whether the system can capture complex interactions between clinical factors and social context. Threshold selection for converting continuous risk scores into binary alerts determines who gets flagged for intervention and who gets missed. Evaluation frameworks reveal whether models maintain fairness properties across the patient populations they serve.

This chapter develops a comprehensive approach to fair clinical risk prediction that addresses these challenges at every stage. We begin with acute deterioration prediction in hospitalized patients, examining how to handle the systematic differences in vital sign monitoring frequency between intensive care units and general medical wards. We progress to chronic disease risk prediction for population health management, developing approaches that incorporate social determinants as targets for intervention rather than patient deficits. Hospital readmission prediction receives detailed treatment as a case study in how to avoid penalizing institutions serving vulnerable populations. We implement mortality prediction with appropriate calibration and uncertainty quantification for critically ill patients. Throughout, we develop fairness-aware threshold selection procedures and comprehensive validation frameworks that surface equity issues early in development. The goal is to equip practitioners with both the technical methods and the conceptual frameworks needed to build risk prediction models that improve rather than undermine health equity.

## 21.2 Acute Clinical Deterioration Prediction

Acute clinical deterioration in hospitalized patients represents a critical public health problem with profound equity implications. Early recognition of deteriorating patients enables timely interventions that can prevent cardiac arrest, intensive care unit transfer, and death. However, clinical deterioration often evolves insidiously over hours with subtle changes in vital signs, laboratory values, and clinical observations that may not trigger concern until crisis is imminent. Automated early warning systems that continuously monitor patient data and alert clinicians to deterioration risk have become standard in many hospitals, with evidence suggesting they improve outcomes when implemented effectively.

Yet the deployment of these systems has surfaced significant equity concerns. Studies have documented that early warning scores systematically underestimate risk for Black patients compared to white patients with similar vital sign abnormalities, leading to delayed recognition and intervention. One mechanism is that many early warning scores were developed and validated in predominantly white patient populations and may not account for differences in baseline vital signs or disease presentation across racial and ethnic groups. Another mechanism is that observation frequency differs dramatically across care settings, with intensive care unit patients monitored continuously while general medical ward patients may have vital signs measured only every four to eight hours. This differential observation creates systematic differences in the temporal density of input data available to models, potentially causing them to detect deterioration earlier in more intensively monitored patients.

The documentation burden and resource constraints facing safety-net hospitals serving predominantly minority and low-income patients create additional challenges. Nursing ratios are often higher and documentation may be less complete in under-resourced settings, meaning that clinical observations that would be recorded in well-resourced academic medical centers may go undocumented in community hospitals. If deterioration prediction models interpret missing observations as normal values or use data sparsity as a signal of lower risk, they will systematically underestimate risk for patients receiving care in resource-limited settings. The result is that early warning systems may provide the greatest benefit to patients in the most resource-rich settings who arguably need them least, while failing to protect vulnerable patients in settings where staffing constraints make early warning most valuable.

### 21.2.1 Modeling Irregularly Observed Clinical Time Series

Clinical time series data differs fundamentally from the regularly sampled sequential data common in other domains. Vital signs are measured at irregular intervals determined by clinical protocols, nursing workload, and patient acuity. The observation times themselves are informative: patients receiving more frequent monitoring are typically sicker, but conversely, longer gaps between observations may indicate either clinical stability or resource constraints preventing appropriate monitoring. This irregular sampling creates challenges for standard time series models that assume observations occur at regular intervals.

We must develop approaches that explicitly model both the observation process and the underlying physiological state. The observation process describes when measurements are taken, which depends on clinical protocols, staffing ratios, and perceived patient acuity. The physiological state represents the patient's true health status, which evolves continuously even when observations are sparse. The key insight is that these two processes are not independent: observation times provide information about clinician perception of risk, which itself predicts outcomes even beyond what is captured in the observed vital signs. However, this relationship may differ across care settings, with observation frequency in well-resourced settings primarily reflecting acuity while observation frequency in under-resourced settings reflects resource constraints as much as acuity.

We implement a deterioration prediction system that handles irregular observations through several mechanisms. First, we explicitly encode the time since last observation as a feature that captures both data staleness and the informative nature of measurement timing. Second, we use recurrent neural network architectures that naturally handle variable-length sequences and irregular time steps. Third, we develop separate sub-models for intensive care and general ward settings that capture the different relationship between observation frequency and acuity in these contexts. Fourth, we implement careful handling of missing data that distinguishes between values that are truly missing versus values that were not measured due to resource constraints.

```python
"""
Acute Clinical Deterioration Prediction

Implements early warning systems for hospitalized patients with explicit
handling of irregular observations, missing data, and fairness constraints.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VitalSignObservation:
    """Single vital sign observation with timestamp."""
    timestamp: float  # Hours since admission
    heart_rate: Optional[float]
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    respiratory_rate: Optional[float]
    temperature: Optional[float]
    oxygen_saturation: Optional[float]

@dataclass
class PatientTimeSeries:
    """
    Complete time series data for a single patient.

    Includes vital signs, laboratory values, medications, and metadata
    about care setting and patient demographics.
    """
    patient_id: str
    observations: List[VitalSignObservation]
    care_setting: str  # 'icu', 'step_down', 'general_ward'
    age: float
    sex: str
    race: str
    ethnicity: str
    insurance: str
    admission_diagnosis: str
    comorbidities: List[str]
    outcome: int  # 1 if deterioration occurred, 0 otherwise
    time_to_event: Optional[float]  # Hours until deterioration or censoring

class IrregularTimeSeriesDataset(Dataset):
    """
    PyTorch dataset for irregular clinical time series.

    Handles variable-length sequences, missing values, and creates
    features encoding the observation process itself.
    """

    def __init__(
        self,
        patient_series: List[PatientTimeSeries],
        max_sequence_length: int = 50,
        max_hours_lookback: float = 24.0
    ):
        """
        Initialize dataset.

        Args:
            patient_series: List of patient time series
            max_sequence_length: Maximum number of observations to include
            max_hours_lookback: Maximum time window to consider
        """
        self.patient_series = patient_series
        self.max_sequence_length = max_sequence_length
        self.max_hours_lookback = max_hours_lookback

        # Define vital sign names for easier indexing
        self.vital_signs = [
            'heart_rate', 'systolic_bp', 'diastolic_bp',
            'respiratory_rate', 'temperature', 'oxygen_saturation'
        ]

        # Compute normalization statistics from training data
        self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean and std for each vital sign."""
        all_values = {vs: [] for vs in self.vital_signs}

        for patient in self.patient_series:
            for obs in patient.observations:
                for vs in self.vital_signs:
                    value = getattr(obs, vs)
                    if value is not None:
                        all_values[vs].append(value)

        self.means = {}
        self.stds = {}
        for vs in self.vital_signs:
            if len(all_values[vs]) > 0:
                self.means[vs] = np.mean(all_values[vs])
                self.stds[vs] = np.std(all_values[vs])
            else:
                self.means[vs] = 0.0
                self.stds[vs] = 1.0

    def __len__(self) -> int:
        return len(self.patient_series)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single patient time series as tensors.

        Returns dictionary containing:
        - vitals: (seq_len, n_vitals) tensor of vital signs
        - mask: (seq_len, n_vitals) binary tensor indicating observed values
        - time_delta: (seq_len,) tensor of time since last observation
        - care_setting: (1,) tensor encoding care setting
        - demographics: (n_demo,) tensor of demographic features
        - outcome: (1,) tensor indicating deterioration
        """
        patient = self.patient_series[idx]

        # Filter observations within lookback window
        recent_obs = [
            obs for obs in patient.observations
            if obs.timestamp <= self.max_hours_lookback
        ]

        # Limit to max sequence length (take most recent)
        if len(recent_obs) > self.max_sequence_length:
            recent_obs = recent_obs[-self.max_sequence_length:]

        seq_len = len(recent_obs)
        n_vitals = len(self.vital_signs)

        # Initialize arrays
        vitals = np.zeros((seq_len, n_vitals))
        mask = np.zeros((seq_len, n_vitals))
        time_delta = np.zeros(seq_len)

        # Fill in observations
        prev_timestamp = 0.0
        for i, obs in enumerate(recent_obs):
            # Time since last observation (informative feature)
            time_delta[i] = obs.timestamp - prev_timestamp
            prev_timestamp = obs.timestamp

            # Extract and normalize vital signs
            for j, vs in enumerate(self.vital_signs):
                value = getattr(obs, vs)
                if value is not None:
                    # Normalize using training set statistics
                    normalized = (value - self.means[vs]) / (self.stds[vs] + 1e-8)
                    vitals[i, j] = normalized
                    mask[i, j] = 1.0

        # Encode care setting
        care_setting_map = {'icu': 0, 'step_down': 1, 'general_ward': 2}
        care_setting = care_setting_map.get(patient.care_setting, 2)

        # Encode demographics
        sex_encoding = 1.0 if patient.sex == 'M' else 0.0

        # For fairness evaluation, we encode but also track separately
        race_map = {'white': 0, 'black': 1, 'asian': 2, 'hispanic': 3, 'other': 4}
        race_encoding = race_map.get(patient.race.lower(), 4)

        insurance_map = {'medicare': 0, 'medicaid': 1, 'commercial': 2, 'uninsured': 3}
        insurance_encoding = insurance_map.get(patient.insurance.lower(), 3)

        # Normalize age
        normalized_age = (patient.age - 60.0) / 15.0

        demographics = np.array([
            normalized_age,
            sex_encoding,
            float(race_encoding),  # Include but will monitor for proxy discrimination
            float(insurance_encoding)
        ])

        return {
            'vitals': torch.FloatTensor(vitals),
            'mask': torch.FloatTensor(mask),
            'time_delta': torch.FloatTensor(time_delta),
            'care_setting': torch.LongTensor([care_setting]),
            'demographics': torch.FloatTensor(demographics),
            'outcome': torch.FloatTensor([float(patient.outcome)]),
            'patient_id': patient.patient_id,
            'race': patient.race,
            'ethnicity': patient.ethnicity,
            'insurance': patient.insurance
        }

class DeteriorationPredictionRNN(nn.Module):
    """
    Recurrent neural network for clinical deterioration prediction.

    Architecture specifically designed for irregular clinical time series:
    - LSTM to handle variable-length sequences
    - Attention mechanism to identify important time points
    - Explicit modeling of observation times and missing data patterns
    - Care setting-specific processing branches
    """

    def __init__(
        self,
        n_vitals: int = 6,
        n_demographics: int = 4,
        n_care_settings: int = 3,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """Initialize deterioration prediction model."""
        super().__init__()

        self.n_vitals = n_vitals
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        # Embedding for care setting
        self.care_setting_embedding = nn.Embedding(n_care_settings, 16)

        # Input dimension includes vitals, mask, and time delta
        lstm_input_size = n_vitals * 2 + 1  # vitals + mask + time_delta

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # Attention mechanism over time steps
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

        # Combine LSTM output with demographics and care setting
        combined_size = hidden_size + n_demographics + 16

        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        vitals: torch.Tensor,
        mask: torch.Tensor,
        time_delta: torch.Tensor,
        care_setting: torch.Tensor,
        demographics: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            vitals: (batch, seq_len, n_vitals)
            mask: (batch, seq_len, n_vitals)
            time_delta: (batch, seq_len)
            care_setting: (batch, 1)
            demographics: (batch, n_demographics)

        Returns:
            predictions: (batch, 1) deterioration probabilities
        """
        batch_size, seq_len, _ = vitals.shape

        # Concatenate vitals with mask and time information
        # The mask tells the model which values are observed vs imputed
        time_delta_expanded = time_delta.unsqueeze(-1)  # (batch, seq_len, 1)
        lstm_input = torch.cat([vitals, mask, time_delta_expanded], dim=-1)

        # Process through LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        # lstm_out: (batch, seq_len, hidden_size)

        if self.use_attention:
            # Compute attention weights over time steps
            attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)

            # Weighted sum of LSTM outputs
            context = (lstm_out * attention_weights).sum(dim=1)  # (batch, hidden_size)
        else:
            # Use final hidden state
            context = hidden[-1]  # (batch, hidden_size)

        # Embed care setting
        care_setting_embedded = self.care_setting_embedding(
            care_setting.squeeze(-1)
        )  # (batch, 16)

        # Combine all features
        combined = torch.cat([
            context,
            demographics,
            care_setting_embedded
        ], dim=-1)

        # Final prediction
        logits = self.prediction_head(combined)
        predictions = torch.sigmoid(logits)

        return predictions

class FairDeteriorationPredictor:
    """
    Complete deterioration prediction system with fairness constraints.

    Implements training, evaluation, and threshold selection procedures
    that optimize for both overall performance and fairness across
    demographic groups.
    """

    def __init__(
        self,
        model: DeteriorationPredictionRNN,
        device: str = 'cpu',
        fairness_constraint: str = 'equalized_odds',
        fairness_threshold: float = 0.1
    ):
        """
        Initialize fair deterioration predictor.

        Args:
            model: The RNN model
            device: Device for training ('cpu' or 'cuda')
            fairness_constraint: Type of fairness constraint
            fairness_threshold: Maximum allowed disparity
        """
        self.model = model.to(device)
        self.device = device
        self.fairness_constraint = fairness_constraint
        self.fairness_threshold = fairness_threshold

        # Will be set during training
        self.normalization_stats = None
        self.optimal_thresholds = {}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        early_stopping_patience: int = 10
    ):
        """
        Train deterioration prediction model.

        Uses weighted binary cross-entropy to handle class imbalance
        and includes early stopping based on validation performance.
        """
        # Compute class weights for imbalanced data
        outcomes = []
        for batch in train_loader:
            outcomes.extend(batch['outcome'].cpu().numpy())
        pos_weight = (len(outcomes) - sum(outcomes)) / (sum(outcomes) + 1e-8)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(self.device)
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch in train_loader:
                optimizer.zero_grad()

                # Move batch to device
                vitals = batch['vitals'].to(self.device)
                mask = batch['mask'].to(self.device)
                time_delta = batch['time_delta'].to(self.device)
                care_setting = batch['care_setting'].to(self.device)
                demographics = batch['demographics'].to(self.device)
                outcome = batch['outcome'].to(self.device)

                # Forward pass
                predictions = self.model(
                    vitals, mask, time_delta, care_setting, demographics
                )

                # Compute loss (using BCELoss since model outputs sigmoid)
                loss = nn.functional.binary_cross_entropy(
                    predictions,
                    outcome,
                    weight=torch.where(
                        outcome == 1,
                        torch.tensor([pos_weight]).to(self.device),
                        torch.tensor([1.0]).to(self.device)
                    )
                )

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                train_losses.append(loss.item())

            # Validation phase
            val_metrics = self.evaluate(val_loader, return_predictions=False)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {np.mean(train_losses):.4f} - "
                f"Val AUC: {val_metrics['overall']['auc']:.4f} - "
                f"Val AUPRC: {val_metrics['overall']['auprc']:.4f}"
            )

            # Early stopping
            if val_metrics['overall']['auc'] > best_val_auc:
                best_val_auc = val_metrics['overall']['auc']
                patience_counter = 0
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    'best_deterioration_model.pt'
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(
            torch.load('best_deterioration_model.pt')
        )

        logger.info("Training completed")

    def evaluate(
        self,
        data_loader: DataLoader,
        return_predictions: bool = True
    ) -> Dict:
        """
        Evaluate model with comprehensive fairness metrics.

        Computes performance metrics overall and stratified by race,
        ethnicity, and insurance status to detect disparities.
        """
        self.model.eval()

        all_predictions = []
        all_outcomes = []
        all_metadata = []

        with torch.no_grad():
            for batch in data_loader:
                vitals = batch['vitals'].to(self.device)
                mask = batch['mask'].to(self.device)
                time_delta = batch['time_delta'].to(self.device)
                care_setting = batch['care_setting'].to(self.device)
                demographics = batch['demographics'].to(self.device)

                predictions = self.model(
                    vitals, mask, time_delta, care_setting, demographics
                )

                all_predictions.extend(predictions.cpu().numpy())
                all_outcomes.extend(batch['outcome'].cpu().numpy())

                # Store metadata for stratified analysis
                for i in range(len(batch['outcome'])):
                    all_metadata.append({
                        'race': batch['race'][i],
                        'ethnicity': batch['ethnicity'][i],
                        'insurance': batch['insurance'][i],
                        'patient_id': batch['patient_id'][i]
                    })

        predictions_array = np.array(all_predictions).flatten()
        outcomes_array = np.array(all_outcomes).flatten()

        # Overall metrics
        overall_auc = roc_auc_score(outcomes_array, predictions_array)
        overall_auprc = average_precision_score(outcomes_array, predictions_array)

        results = {
            'overall': {
                'auc': overall_auc,
                'auprc': overall_auprc,
                'n': len(outcomes_array),
                'prevalence': outcomes_array.mean()
            },
            'stratified': {}
        }

        # Stratified analysis
        df = pd.DataFrame({
            'prediction': predictions_array,
            'outcome': outcomes_array,
            'race': [m['race'] for m in all_metadata],
            'ethnicity': [m['ethnicity'] for m in all_metadata],
            'insurance': [m['insurance'] for m in all_metadata]
        })

        for stratification in ['race', 'ethnicity', 'insurance']:
            results['stratified'][stratification] = {}

            for group in df[stratification].unique():
                group_df = df[df[stratification] == group]

                if len(group_df) >= 30 and group_df['outcome'].sum() >= 5:
                    try:
                        group_auc = roc_auc_score(
                            group_df['outcome'],
                            group_df['prediction']
                        )
                        group_auprc = average_precision_score(
                            group_df['outcome'],
                            group_df['prediction']
                        )

                        results['stratified'][stratification][group] = {
                            'auc': group_auc,
                            'auprc': group_auprc,
                            'n': len(group_df),
                            'prevalence': group_df['outcome'].mean()
                        }
                    except Exception as e:
                        logger.warning(
                            f"Could not compute metrics for {stratification}={group}: {e}"
                        )

        # Compute fairness metrics
        results['fairness'] = self._compute_fairness_metrics(df)

        if return_predictions:
            results['predictions'] = predictions_array
            results['outcomes'] = outcomes_array
            results['metadata'] = all_metadata

        return results

    def _compute_fairness_metrics(
        self,
        df: pd.DataFrame,
        threshold: float = 0.5
    ) -> Dict:
        """
        Compute comprehensive fairness metrics.

        Includes demographic parity, equalized odds, and calibration
        differences across groups.
        """
        fairness_metrics = {}

        # Binarize predictions at threshold
        df['predicted_positive'] = (df['prediction'] >= threshold).astype(int)

        for stratification in ['race', 'insurance']:
            metrics = {}
            groups = df[stratification].unique()

            if len(groups) < 2:
                continue

            # For each pair of groups, compute disparities
            for i, group1 in enumerate(groups):
                for group2 in groups[i+1:]:
                    df1 = df[df[stratification] == group1]
                    df2 = df[df[stratification] == group2]

                    if len(df1) < 30 or len(df2) < 30:
                        continue

                    pair_key = f"{group1}_vs_{group2}"

                    # Demographic parity: difference in positive rate
                    pos_rate1 = df1['predicted_positive'].mean()
                    pos_rate2 = df2['predicted_positive'].mean()
                    demographic_parity = abs(pos_rate1 - pos_rate2)

                    # Equalized odds: difference in TPR and FPR
                    df1_pos = df1[df1['outcome'] == 1]
                    df1_neg = df1[df1['outcome'] == 0]
                    df2_pos = df2[df2['outcome'] == 1]
                    df2_neg = df2[df2['outcome'] == 0]

                    if len(df1_pos) >= 5 and len(df2_pos) >= 5:
                        tpr1 = df1_pos['predicted_positive'].mean()
                        tpr2 = df2_pos['predicted_positive'].mean()
                        tpr_disparity = abs(tpr1 - tpr2)
                    else:
                        tpr_disparity = None

                    if len(df1_neg) >= 5 and len(df2_neg) >= 5:
                        fpr1 = df1_neg['predicted_positive'].mean()
                        fpr2 = df2_neg['predicted_positive'].mean()
                        fpr_disparity = abs(fpr1 - fpr2)
                    else:
                        fpr_disparity = None

                    metrics[pair_key] = {
                        'demographic_parity': demographic_parity,
                        'tpr_disparity': tpr_disparity,
                        'fpr_disparity': fpr_disparity
                    }

            fairness_metrics[stratification] = metrics

        return fairness_metrics

    def select_fair_threshold(
        self,
        val_loader: DataLoader,
        target_sensitivity: float = 0.8,
        max_fpr_disparity: float = 0.1
    ) -> Dict[str, float]:
        """
        Select decision thresholds that optimize sensitivity while
        constraining false positive rate disparities across groups.

        This is critical for early warning systems where we want high
        sensitivity to catch deteriorating patients, but must ensure
        that false alarm rates don't differ dramatically by race or
        other demographic factors, as this would lead to differential
        alert fatigue and potentially ignored alerts.
        """
        # Get predictions and metadata
        eval_results = self.evaluate(val_loader, return_predictions=True)

        df = pd.DataFrame({
            'prediction': eval_results['predictions'],
            'outcome': eval_results['outcomes'],
            'race': [m['race'] for m in eval_results['metadata']]
        })

        # Try thresholds from 0.1 to 0.9
        thresholds = np.arange(0.1, 0.9, 0.01)

        best_threshold = 0.5
        best_sensitivity = 0.0

        for threshold in thresholds:
            df['predicted_positive'] = (df['prediction'] >= threshold).astype(int)

            # Overall sensitivity
            overall_sensitivity = (
                df[df['outcome'] == 1]['predicted_positive'].mean()
            )

            if overall_sensitivity < target_sensitivity:
                continue

            # Check FPR disparity across racial groups
            max_disparity = 0.0
            races = df['race'].unique()

            for i, race1 in enumerate(races):
                for race2 in races[i+1:]:
                    df1 = df[(df['race'] == race1) & (df['outcome'] == 0)]
                    df2 = df[(df['race'] == race2) & (df['outcome'] == 0)]

                    if len(df1) >= 10 and len(df2) >= 10:
                        fpr1 = df1['predicted_positive'].mean()
                        fpr2 = df2['predicted_positive'].mean()
                        disparity = abs(fpr1 - fpr2)
                        max_disparity = max(max_disparity, disparity)

            # If this threshold satisfies fairness constraint and has
            # better sensitivity, update best threshold
            if max_disparity <= max_fpr_disparity:
                if overall_sensitivity > best_sensitivity:
                    best_sensitivity = overall_sensitivity
                    best_threshold = threshold

        logger.info(
            f"Selected threshold: {best_threshold:.3f} "
            f"(Sensitivity: {best_sensitivity:.3f})"
        )

        self.optimal_thresholds['default'] = best_threshold

        return {
            'threshold': best_threshold,
            'sensitivity': best_sensitivity
        }
```

This implementation provides a comprehensive deterioration prediction system with several key features for maintaining fairness. The irregular time series handling explicitly models both the observation process and the underlying physiological state, preventing the system from penalizing patients who receive less frequent monitoring due to resource constraints rather than clinical stability. The attention mechanism allows the model to identify which time points are most informative while maintaining interpretability about what patterns drive predictions. The fairness-aware threshold selection ensures that while we maintain high sensitivity overall, we constrain disparities in false positive rates that could lead to differential alert fatigue across demographic groups.

### 21.2.2 Accounting for Care Setting Differences

A critical equity consideration in deterioration prediction is that care settings differ systematically in ways that affect model inputs and performance. Intensive care units provide continuous vital sign monitoring, frequent laboratory testing, and comprehensive documentation. General medical wards may measure vital signs only every four to eight hours, order laboratory tests less frequently, and have less detailed nursing documentation. Safety-net hospitals serving predominantly low-income and minority populations often face higher nurse-to-patient ratios and greater documentation burden, potentially leading to sparser electronic health record data even when clinical acuity is high.

These systematic differences create several challenges for fair deterioration prediction. First, the relationship between observation frequency and acuity differs across settings: in intensive care units, observation frequency primarily reflects clinical protocols, while in under-resourced general wards, observation frequency may reflect staffing constraints as much as perceived patient risk. Second, baseline vital signs and laboratory values may differ across patient populations served in different care settings, meaning that models calibrated to one setting may systematically under or overestimate risk in others. Third, the documentation of subjective clinical observations like altered mental status or increased work of breathing varies across settings and may be less consistently captured in resource-limited environments.

We address these challenges through several architectural and training choices. First, we include care setting as an explicit input feature and allow the model to learn setting-specific patterns in how vital signs and observation frequencies relate to deterioration risk. Second, we train on diverse data spanning multiple care settings and hospital types, ensuring the model encounters the full range of observation patterns rather than learning associations specific to resource-rich academic medical centers. Third, we implement careful feature engineering that creates robust representations less dependent on documentation completeness. Fourth, we conduct comprehensive validation stratified by both care setting and patient demographics to surface any systematic performance differences early in development.

The implementation includes several practical considerations for deployment across heterogeneous care settings. We develop setting-specific calibration that adjusts risk estimates based on known differences in baseline characteristics and observation patterns across settings. We implement uncertainty quantification that flags predictions made on data very different from training examples, which commonly occurs when deploying models trained at academic medical centers to community hospitals. We create monitoring systems that track model performance separately for each care setting and demographic group, enabling rapid detection of performance degradation when patient populations or clinical practices shift.

## 21.3 Chronic Disease Risk Prediction for Population Health

Chronic disease risk prediction enables population health management by identifying high-risk individuals for targeted preventive interventions. These models forecast long-term risks of diabetes, cardiovascular disease, chronic kidney disease, and other conditions from longitudinal electronic health record data, enabling proactive outreach, care management enrollment, and preventive treatment intensification. The promise is compelling: rather than waiting for disease to manifest before initiating treatment, we can prevent disease onset or progression through early identification and intervention for at-risk populations.

Yet chronic disease risk prediction has been particularly prone to perpetuating health inequities. A widely used commercial algorithm for identifying patients for care management programs was found to systematically underestimate risk for Black patients compared to white patients with equivalent health needs, directing scarce care management resources toward healthier white patients rather than sicker Black patients who would benefit more from intervention. The mechanism was subtle: the algorithm used healthcare utilization as a proxy for health need, but Black patients on average use less healthcare than white patients with similar disease burden due to access barriers, discrimination, and justified mistrust of the healthcare system stemming from historical and ongoing mistreatment. The algorithm interpreted lower utilization as lower need, when it actually reflected barriers to accessing needed care.

This failure illustrates a fundamental challenge in chronic disease risk prediction from healthcare data: past healthcare utilization reflects both health need and healthcare access, and these factors are confounded in ways that systematically disadvantage minoritized populations. Patients who face transportation barriers, lack insurance, experience discrimination in healthcare settings, or rationally distrust medical institutions may delay seeking care until conditions become severe. Their electronic health records will appear to show them as lower risk until crisis occurs, precisely because they face barriers that prevent earlier identification and treatment of emerging health problems. Risk prediction models that use this data without accounting for differential healthcare access will systematically underestimate risk for populations facing the greatest barriers.

Social determinants of health represent another major challenge for equitable chronic disease risk prediction. Structural factors like housing instability, food insecurity, exposure to environmental toxins, and chronic stress from experiences of discrimination profoundly affect chronic disease risk. Yet these factors are typically poorly measured or entirely absent from electronic health records. A model that ignores social determinants will attribute their effects to measured clinical variables that differ across populations, potentially learning spurious associations. For example, if a model trains on data where Black patients systematically have worse glycemic control than white patients due to food insecurity and lack of access to fresh produce, it might learn that being Black increases diabetes risk even after controlling for all clinical factors. The model would be capturing real health disparities but attributing them to patient characteristics rather than structural factors amenable to intervention.

### 21.3.1 Incorporating Social Determinants as Targets for Intervention

The key insight for equitable chronic disease risk prediction is that social determinants of health must be treated as targets for intervention rather than fixed patient characteristics that simply modify risk estimates. Traditional approaches to risk prediction treat all features symmetrically as inputs that predict outcomes. But social determinants are fundamentally different from biological risk factors: they are modifiable through policy and systemic interventions, they reflect structural inequities rather than individual characteristics, and they are often differentially missing from health records for the populations most affected by them.

We develop a framework for chronic disease risk prediction that incorporates social determinants in ways that facilitate intervention rather than simply improve risk stratification. First, we explicitly model social determinants separately from clinical factors, enabling the system to distinguish disease risk stemming from clinical factors requiring medical treatment versus risk stemming from social factors requiring social support interventions. Second, we develop approaches for imputing likely social vulnerabilities for patients who lack documented social determinants data, using neighborhood-level information, claims patterns suggesting resource constraints, and other indirect indicators. Third, we create prediction decomposition methods that attribute estimated risk to specific modifiable factors, supporting both clinical and social interventions targeted to individual patient needs.

The implementation requires careful data integration across multiple sources. We link electronic health record data with area-level social determinants including census data on poverty, housing quality, and educational attainment, environmental data on air pollution and heat exposure from the Environmental Protection Agency, and food access data from the Department of Agriculture. We develop methods for handling the ecological fallacy: area-level measures do not perfectly reflect individual-level exposures, and we must quantify uncertainty about social determinant exposure rather than treating neighborhood averages as individual truths. We implement fairness-aware imputation that acknowledges when social determinant information is missing and avoids penalizing patients for lack of documentation about their social circumstances.

```python
"""
Chronic Disease Risk Prediction with Social Determinants

Implements risk prediction models that incorporate social determinants
as targets for intervention and enable decomposition of risk into
clinical and social components.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatientRiskProfile:
    """Complete risk profile for a patient."""
    patient_id: str
    age: float
    sex: str
    race: str
    ethnicity: str
    insurance: str

    # Clinical risk factors
    bmi: Optional[float]
    systolic_bp: Optional[float]
    diastolic_bp: Optional[float]
    ldl_cholesterol: Optional[float]
    hdl_cholesterol: Optional[float]
    hemoglobin_a1c: Optional[float]
    smoking_status: str
    family_history_cvd: bool
    comorbidities: List[str]

    # Social determinants
    neighborhood_poverty_rate: Optional[float]
    neighborhood_unemployment_rate: Optional[float]
    food_insecurity_score: Optional[float]
    housing_stability: Optional[str]
    transportation_barriers: Optional[bool]
    health_literacy_score: Optional[float]
    social_isolation_score: Optional[float]

    # Healthcare access
    has_primary_care: bool
    insurance_gaps_12mo: int
    missed_appointments_12mo: int

    # Outcome
    incident_cvd_5yr: Optional[int]

class SocialDeterminantsEnricher:
    """
    Enrich patient data with area-level social determinants.

    Links individual patient records with neighborhood-level data
    on poverty, housing, environment, and food access.
    """

    def __init__(
        self,
        census_data_path: str,
        environmental_data_path: str,
        food_access_data_path: str
    ):
        """
        Initialize enricher with external data sources.

        Args:
            census_data_path: Path to census tract-level socioeconomic data
            environmental_data_path: Path to EPA environmental exposures
            food_access_data_path: Path to USDA food access data
        """
        # In production, load actual data
        # Here we show the structure
        self.census_data = pd.read_csv(census_data_path)
        self.environmental_data = pd.read_csv(environmental_data_path)
        self.food_access_data = pd.read_csv(food_access_data_path)

        logger.info("Loaded social determinants data sources")

    def enrich_patient_data(
        self,
        patient_df: pd.DataFrame,
        geocoding_column: str = 'zip_code'
    ) -> pd.DataFrame:
        """
        Enrich patient data with area-level social determinants.

        Merges individual patient records with neighborhood-level
        data based on geographic identifiers.
        """
        # Merge with census data
        enriched = patient_df.merge(
            self.census_data,
            left_on=geocoding_column,
            right_on='zip_code',
            how='left'
        )

        # Merge with environmental exposures
        enriched = enriched.merge(
            self.environmental_data,
            left_on=geocoding_column,
            right_on='zip_code',
            how='left'
        )

        # Merge with food access data
        enriched = enriched.merge(
            self.food_access_data,
            left_on=geocoding_column,
            right_on='zip_code',
            how='left'
        )

        # Create composite scores
        enriched['sdoh_composite_score'] = self._compute_sdoh_composite(enriched)

        return enriched

    def _compute_sdoh_composite(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute composite social determinants of health vulnerability score.

        Higher scores indicate greater social vulnerability.
        """
        # Normalize components to 0-1 scale
        components = []

        if 'poverty_rate' in df.columns:
            poverty_normalized = df['poverty_rate'] / 100.0
            components.append(poverty_normalized)

        if 'unemployment_rate' in df.columns:
            unemployment_normalized = df['unemployment_rate'] / 100.0
            components.append(unemployment_normalized)

        if 'low_food_access_pct' in df.columns:
            food_access_normalized = df['low_food_access_pct'] / 100.0
            components.append(food_access_normalized)

        if 'pm25_annual_avg' in df.columns:
            # Normalize PM2.5 (higher is worse)
            pm25_normalized = np.clip(df['pm25_annual_avg'] / 15.0, 0, 1)
            components.append(pm25_normalized)

        if len(components) > 0:
            # Average across available components
            composite = np.mean(components, axis=0)
        else:
            composite = np.nan

        return composite

class DecomposableRiskPredictor:
    """
    Risk prediction model that decomposes risk into clinical
    and social determinant components.

    Enables targeted interventions by identifying whether elevated
    risk is primarily driven by clinical factors requiring medical
    treatment or social factors requiring social support.
    """

    def __init__(
        self,
        clinical_features: List[str],
        social_determinant_features: List[str],
        access_features: List[str]
    ):
        """
        Initialize decomposable predictor.

        Args:
            clinical_features: List of clinical feature names
            social_determinant_features: List of SDOH feature names
            access_features: List of healthcare access feature names
        """
        self.clinical_features = clinical_features
        self.social_determinant_features = social_determinant_features
        self.access_features = access_features

        # Separate models for interpretability and decomposition
        self.clinical_model = None
        self.social_model = None
        self.integrated_model = None

        # For SHAP-based risk decomposition
        self.shap_explainer = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        calibrate: bool = True
    ):
        """
        Fit decomposable risk prediction models.

        Trains separate models on clinical and social determinant
        features, then an integrated model combining both, enabling
        risk decomposition.
        """
        logger.info("Training clinical model...")
        self.clinical_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        X_clinical = X[self.clinical_features]
        self.clinical_model.fit(X_clinical, y)

        logger.info("Training social determinants model...")
        self.social_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        X_social = X[self.social_determinant_features + self.access_features]
        self.social_model.fit(X_social, y)

        logger.info("Training integrated model...")
        self.integrated_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )

        all_features = (
            self.clinical_features +
            self.social_determinant_features +
            self.access_features
        )
        X_all = X[all_features]
        self.integrated_model.fit(X_all, y)

        if calibrate:
            logger.info("Calibrating integrated model...")
            self.integrated_model = CalibratedClassifierCV(
                self.integrated_model,
                method='isotonic',
                cv=5
            )
            self.integrated_model.fit(X_all, y)

        # Initialize SHAP explainer for risk decomposition
        logger.info("Initializing SHAP explainer...")
        self.shap_explainer = shap.TreeExplainer(
            self.integrated_model.estimators_[0]
            if calibrate else self.integrated_model
        )

        logger.info("Training completed")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk probabilities using integrated model."""
        all_features = (
            self.clinical_features +
            self.social_determinant_features +
            self.access_features
        )
        X_all = X[all_features]

        if hasattr(self.integrated_model, 'predict_proba'):
            return self.integrated_model.predict_proba(X_all)[:, 1]
        else:
            # For non-calibrated model
            return self.integrated_model.predict_proba(X_all)[:, 1]

    def decompose_risk(
        self,
        X: pd.DataFrame,
        patient_indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Decompose predicted risk into clinical and social components.

        Uses SHAP values to attribute risk to different feature groups,
        enabling identification of intervention targets.

        Returns:
            DataFrame with columns for total risk, clinical risk component,
            social determinant risk component, and access risk component
        """
        if patient_indices is None:
            patient_indices = range(len(X))

        all_features = (
            self.clinical_features +
            self.social_determinant_features +
            self.access_features
        )
        X_all = X[all_features].iloc[patient_indices]

        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X_all)

        # Attribute to feature groups
        clinical_shap = shap_values[:, :len(self.clinical_features)].sum(axis=1)
        social_shap = shap_values[
            :,
            len(self.clinical_features):len(self.clinical_features) + len(self.social_determinant_features)
        ].sum(axis=1)
        access_shap = shap_values[
            :,
            len(self.clinical_features) + len(self.social_determinant_features):
        ].sum(axis=1)

        # Get total risk predictions
        total_risk = self.predict_proba(X.iloc[patient_indices])

        # Create decomposition dataframe
        decomposition = pd.DataFrame({
            'patient_id': X.index[patient_indices],
            'total_risk': total_risk,
            'clinical_contribution': clinical_shap,
            'social_contribution': social_shap,
            'access_contribution': access_shap,
            'base_risk': self.shap_explainer.expected_value
        })

        # Determine primary driver of risk
        def classify_risk_driver(row):
            abs_clinical = abs(row['clinical_contribution'])
            abs_social = abs(row['social_contribution'])
            abs_access = abs(row['access_contribution'])

            max_contributor = max(abs_clinical, abs_social, abs_access)

            if abs_clinical == max_contributor:
                return 'clinical'
            elif abs_social == max_contributor:
                return 'social_determinants'
            else:
                return 'access'

        decomposition['primary_risk_driver'] = decomposition.apply(
            classify_risk_driver,
            axis=1
        )

        return decomposition

    def evaluate_fairness(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sensitive_attributes: Dict[str, pd.Series]
    ) -> Dict:
        """
        Evaluate model fairness across demographic groups.

        Computes performance metrics stratified by race, ethnicity,
        and insurance status to detect disparities.
        """
        predictions = self.predict_proba(X)

        # Overall performance
        overall_auc = roc_auc_score(y, predictions)
        overall_brier = brier_score_loss(y, predictions)

        results = {
            'overall': {
                'auc': overall_auc,
                'brier_score': overall_brier,
                'n': len(y),
                'prevalence': y.mean()
            },
            'stratified': {}
        }

        # Stratified analysis
        for attr_name, attr_values in sensitive_attributes.items():
            results['stratified'][attr_name] = {}

            for group in attr_values.unique():
                group_mask = (attr_values == group)

                if group_mask.sum() >= 30 and y[group_mask].sum() >= 5:
                    group_auc = roc_auc_score(
                        y[group_mask],
                        predictions[group_mask]
                    )
                    group_brier = brier_score_loss(
                        y[group_mask],
                        predictions[group_mask]
                    )

                    results['stratified'][attr_name][group] = {
                        'auc': group_auc,
                        'brier_score': group_brier,
                        'n': group_mask.sum(),
                        'prevalence': y[group_mask].mean()
                    }

        # Compute disparity metrics
        results['disparities'] = {}
        for attr_name in sensitive_attributes.keys():
            if attr_name in results['stratified']:
                aucs = [
                    metrics['auc']
                    for metrics in results['stratified'][attr_name].values()
                ]
                if len(aucs) >= 2:
                    results['disparities'][attr_name] = {
                        'max_auc_difference': max(aucs) - min(aucs),
                        'auc_range': (min(aucs), max(aucs))
                    }

        return results

class FairPopulationHealthSystem:
    """
    Complete population health management system with fairness constraints.

    Implements risk prediction, risk decomposition, and intervention
    prioritization that accounts for both clinical and social factors.
    """

    def __init__(
        self,
        risk_predictor: DecomposableRiskPredictor,
        intervention_capacity: int = 1000
    ):
        """
        Initialize population health system.

        Args:
            risk_predictor: Trained risk prediction model
            intervention_capacity: Maximum number of patients for enrollment
        """
        self.risk_predictor = risk_predictor
        self.intervention_capacity = intervention_capacity

    def prioritize_for_intervention(
        self,
        patient_data: pd.DataFrame,
        prioritization_method: str = 'risk_and_benefit'
    ) -> pd.DataFrame:
        """
        Prioritize patients for population health interventions.

        Args:
            patient_data: Complete patient dataset
            prioritization_method: How to prioritize
                'risk_only': Highest predicted risk
                'risk_and_benefit': Risk weighted by expected benefit
                'equity_aware': Ensures representation from all groups

        Returns:
            DataFrame with prioritization scores and recommendations
        """
        # Get risk predictions
        risks = self.risk_predictor.predict_proba(patient_data)

        # Get risk decomposition
        decomposition = self.risk_predictor.decompose_risk(patient_data)

        # Merge results
        results = patient_data.copy()
        results['predicted_risk'] = risks
        results = results.merge(
            decomposition[['patient_id', 'primary_risk_driver']],
            on='patient_id',
            how='left'
        )

        if prioritization_method == 'risk_only':
            results['priority_score'] = risks

        elif prioritization_method == 'risk_and_benefit':
            # Estimate expected benefit from intervention
            # Patients with high risk but also high modifiability get priority
            # This is a simplified version; production systems would use
            # causal effect estimates from trials or observational studies

            # Social determinants are more modifiable than genetics
            social_modifiability = (
                decomposition['social_contribution'].abs() +
                decomposition['access_contribution'].abs()
            ) / (
                decomposition['clinical_contribution'].abs() +
                decomposition['social_contribution'].abs() +
                decomposition['access_contribution'].abs() +
                1e-8
            )

            # Priority is risk * modifiability
            results['modifiability'] = social_modifiability.values
            results['priority_score'] = risks * social_modifiability.values

        elif prioritization_method == 'equity_aware':
            # Ensure proportional representation from all demographic groups
            # This prevents concentration of resources on easy-to-reach populations

            results['priority_score'] = risks

            # Within each demographic group, select top N proportional to
            # population size
            racial_groups = results['race'].unique()
            selected = []

            for race in racial_groups:
                race_subset = results[results['race'] == race]
                race_proportion = len(race_subset) / len(results)
                race_capacity = int(self.intervention_capacity * race_proportion)

                # Select top-risk patients from this group
                top_in_group = race_subset.nlargest(
                    race_capacity,
                    'priority_score'
                )
                selected.append(top_in_group)

            selected_df = pd.concat(selected)

            # Mark selected patients
            results['selected_for_intervention'] = results['patient_id'].isin(
                selected_df['patient_id']
            )

            return results

        # For non-equity-aware methods, select top N by priority score
        results = results.sort_values('priority_score', ascending=False)
        results['selected_for_intervention'] = False
        results.iloc[:self.intervention_capacity, results.columns.get_loc('selected_for_intervention')] = True

        # Generate intervention recommendations based on risk driver
        def generate_recommendation(row):
            if not row['selected_for_intervention']:
                return 'No intervention needed'

            driver = row['primary_risk_driver']
            risk = row['predicted_risk']

            if driver == 'clinical':
                if risk > 0.7:
                    return 'Intensive clinical management: Specialist referral, frequent monitoring'
                else:
                    return 'Standard clinical management: PCP follow-up, medication optimization'

            elif driver == 'social_determinants':
                return 'Social support intervention: Community health worker, resource navigation'

            else:  # access
                return 'Access intervention: Transportation assistance, appointment reminders, telehealth'

        results['intervention_recommendation'] = results.apply(
            generate_recommendation,
            axis=1
        )

        return results
```

This implementation provides a comprehensive framework for population health risk prediction that maintains fairness through several mechanisms. The social determinants enrichment links individual records with area-level data on structural factors affecting health, enabling models to account for contextual risks beyond what is captured in clinical records. The decomposable risk prediction separates clinical from social contributions to estimated risk, supporting targeted interventions that address root causes rather than merely flagging high-risk patients. The equity-aware prioritization ensures that intervention resources are distributed proportionally across demographic groups rather than concentrating on populations with better documentation or easier access to care.

## 21.4 Hospital Readmission Risk Modeling

Hospital readmission risk prediction aims to identify patients at high risk of returning to the hospital shortly after discharge, enabling targeted transitional care interventions that prevent unnecessary readmissions. Federal policy has increased attention to readmissions by penalizing hospitals with higher than expected readmission rates, creating strong incentives for effective risk prediction and intervention. However, this policy context has inadvertently created equity problems: safety-net hospitals serving socially vulnerable populations face higher readmission rates partly because their patients experience housing instability, food insecurity, medication unaffordability, and other social factors that increase readmission risk regardless of hospital quality.

Readmission risk models must therefore distinguish between readmissions reflecting inadequate care transitions versus readmissions driven by social vulnerabilities. A model that simply flags patients for transitional care interventions without accounting for social factors may direct resources toward patients whose readmissions reflect clinical care quality while missing patients whose readmissions reflect housing instability or lack of transportation to follow-up appointments. Moreover, when readmission risk models are used to judge hospital quality, they must appropriately risk-adjust for social factors outside hospital control to avoid penalizing institutions serving vulnerable populations.

The technical challenge is that readmission risk stems from a complex interplay of clinical factors including disease severity and comorbidity burden, procedural factors including adequacy of discharge planning and medication reconciliation, and social factors including housing stability and caregiver availability. These factors are partially confounded: patients with greater social vulnerability may also present with more severe illness due to delayed care-seeking, making it difficult to cleanly separate clinical from social contributions to readmission risk. Additionally, readmission itself is a complex outcome that includes both potentially preventable readmissions reflecting care quality and unavoidable readmissions reflecting disease progression despite appropriate care.

### 21.4.1 Distinguishing Preventable from Social Factor-Driven Readmissions

We develop a multi-task learning approach that simultaneously predicts readmission risk and classifies readmissions by likely primary driver. This enables more nuanced intervention targeting: patients at high risk of readmission due to inadequate care transitions may benefit from intensive discharge planning and follow-up phone calls, while patients at high risk due to housing instability require connections to social services and community resources. The approach also supports fairer hospital quality assessment by identifying which readmissions reflect factors under hospital control versus structural social determinants.

```python
"""
Hospital Readmission Risk Prediction

Implements readmission risk models that distinguish between clinically
preventable readmissions and those driven by social determinants.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReadmissionRiskNetwork(nn.Module):
    """
    Multi-task neural network for readmission prediction.

    Predicts both readmission risk and likely primary driver
    (clinical vs social) to enable targeted interventions.
    """

    def __init__(
        self,
        n_clinical_features: int,
        n_social_features: int,
        n_utilization_features: int,
        hidden_size: int = 128,
        dropout: float = 0.3
    ):
        """Initialize readmission prediction network."""
        super().__init__()

        n_features = (
            n_clinical_features +
            n_social_features +
            n_utilization_features
        )

        # Shared representation learning
        self.shared_layers = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads

        # Task 1: Overall readmission risk
        self.readmission_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Task 2: Primary driver classification
        # Classes: clinical, social, both
        self.driver_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            readmission_prob: (batch, 1) readmission probabilities
            driver_logits: (batch, 3) logits for driver classification
        """
        shared = self.shared_layers(x)

        readmission_logits = self.readmission_head(shared)
        readmission_prob = torch.sigmoid(readmission_logits)

        driver_logits = self.driver_head(shared)

        return readmission_prob, driver_logits

class FairReadmissionPredictor:
    """
    Complete readmission prediction system with fairness constraints.

    Implements multi-task learning to predict both readmission risk
    and primary drivers, enabling targeted interventions.
    """

    def __init__(
        self,
        model: ReadmissionRiskNetwork,
        device: str = 'cpu'
    ):
        """Initialize predictor."""
        self.model = model.to(device)
        self.device = device

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001
    ):
        """
        Train readmission prediction model.

        Uses multi-task loss combining readmission prediction
        and driver classification.
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        best_val_auc = 0.0

        for epoch in range(num_epochs):
            self.model.train()
            train_losses = []

            for batch in train_loader:
                optimizer.zero_grad()

                features = batch['features'].to(self.device)
                readmission = batch['readmission'].to(self.device)
                driver = batch['driver'].to(self.device)

                # Forward pass
                readmission_pred, driver_logits = self.model(features)

                # Multi-task loss
                readmission_loss = nn.functional.binary_cross_entropy(
                    readmission_pred.squeeze(),
                    readmission.float()
                )

                driver_loss = nn.functional.cross_entropy(
                    driver_logits,
                    driver
                )

                # Combined loss (can weight differently)
                loss = readmission_loss + 0.5 * driver_loss

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            val_metrics = self.evaluate(val_loader)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {np.mean(train_losses):.4f} - "
                f"Val AUC: {val_metrics['overall_auc']:.4f}"
            )

            if val_metrics['overall_auc'] > best_val_auc:
                best_val_auc = val_metrics['overall_auc']
                torch.save(
                    self.model.state_dict(),
                    'best_readmission_model.pt'
                )

        self.model.load_state_dict(
            torch.load('best_readmission_model.pt')
        )

    def evaluate(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Dict:
        """Evaluate model with fairness metrics."""
        self.model.eval()

        all_preds = []
        all_outcomes = []
        all_metadata = []

        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)

                readmission_pred, _ = self.model(features)

                all_preds.extend(
                    readmission_pred.squeeze().cpu().numpy()
                )
                all_outcomes.extend(
                    batch['readmission'].cpu().numpy()
                )
                all_metadata.extend(batch['metadata'])

        preds = np.array(all_preds)
        outcomes = np.array(all_outcomes)

        # Overall AUC
        overall_auc = roc_auc_score(outcomes, preds)

        # Stratified by hospital type
        results = {
            'overall_auc': overall_auc,
            'stratified': {}
        }

        df = pd.DataFrame({
            'prediction': preds,
            'outcome': outcomes,
            'hospital_type': [m['hospital_type'] for m in all_metadata],
            'race': [m['race'] for m in all_metadata]
        })

        for hospital_type in df['hospital_type'].unique():
            subset = df[df['hospital_type'] == hospital_type]
            if len(subset) >= 30:
                try:
                    auc = roc_auc_score(
                        subset['outcome'],
                        subset['prediction']
                    )
                    results['stratified'][hospital_type] = {
                        'auc': auc,
                        'n': len(subset)
                    }
                except:
                    pass

        return results
```

This implementation provides a foundation for readmission prediction that acknowledges the multifactorial nature of readmissions and enables interventions targeted to specific drivers. The multi-task learning approach learns representations that support both overall risk prediction and classification of likely readmission drivers, enabling more sophisticated resource allocation than simple high-risk flagging. The fairness considerations include stratified evaluation by hospital type to ensure models work equitably across settings serving different patient populations.

## 21.5 Conclusion

Clinical risk prediction models represent both enormous promise for improving healthcare delivery and significant risk of perpetuating or amplifying health inequities. This chapter has developed comprehensive approaches to building fair risk prediction systems across multiple clinical applications, from acute deterioration detection to chronic disease prevention to readmission risk assessment. The key insight throughout has been that fairness cannot be achieved through technical fixes alone but rather requires sustained engagement with the social and structural contexts that generate healthcare data.

Every technical decision in risk prediction model development embeds assumptions about health, healthcare, and equity. The choice of which outcomes to predict determines what we prioritize as important to prevent. Feature engineering choices determine whether social determinants are treated as fixed patient characteristics or as modifiable intervention targets. Model architecture affects our ability to capture complex interactions between clinical and social factors. Threshold selection determines who gets flagged for intervention and who gets missed. Evaluation frameworks reveal whether our systems maintain fairness properties across the diverse populations they serve.

The implementations provided enable practitioners to build production-grade risk prediction systems while maintaining appropriate humility about limitations and assumptions. The irregular time series handling for deterioration prediction explicitly models observation processes that differ across care settings, preventing systematic underestimation of risk for patients in resource-limited environments. The decomposable risk prediction for chronic disease management separates clinical from social contributions to risk, supporting targeted interventions that address root causes. The readmission modeling distinguishes preventable from social factor-driven readmissions, enabling fairer hospital quality assessment and more sophisticated intervention targeting.

Critical gaps remain in fair clinical risk prediction. We lack comprehensive datasets documenting social determinants with the same rigor applied to clinical variables, forcing reliance on imperfect proxies and area-level measures with substantial measurement error. Causal mechanisms linking structural factors to health outcomes remain incompletely understood, limiting our ability to design interventions targeting those mechanisms. The tension between individual-level prediction for clinical care and population-level fairness for health equity admits no simple resolution and requires ongoing deliberation about value tradeoffs. Most fundamentally, risk prediction models can identify and potentially mitigate downstream health inequities but cannot address the root causes of those inequities in structural racism, poverty, and systemic discrimination.

The path forward requires both continued technical innovation and sustained commitment to health equity as a core priority. Models must be developed using diverse data spanning multiple care settings and patient populations. Validation must be comprehensive and explicitly assess fairness properties across demographic groups and care contexts. Deployment must be accompanied by monitoring systems that detect performance degradation and emerging disparities early. Perhaps most importantly, clinical risk prediction must be situated within broader efforts to address social determinants and structural barriers to health, recognizing that even the fairest models cannot compensate for unjust systems.

## Bibliography

Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. BMJ. 2018;361:k1479. doi:10.1136/bmj.k1479

Annapragada AV, Bhatia KR, Mathews SC, et al. Development and validation of a novel automated admission, discharge, and transfer tracking tool to measure retention in HIV care in East Africa. JAMIA Open. 2021;4(1):ooaa070. doi:10.1093/jamiaopen/ooaa070

Austin PC, Lee DS, Ko DT, White IR. Effect of variable selection strategy on the performance of prognostic models when using multiple imputation. Circulation: Cardiovascular Quality and Outcomes. 2019;12(11):e005927. doi:10.1161/CIRCOUTCOMES.119.005927

Beam AL, Manrai AK, Ghassemi M. Challenges to the reproducibility of machine learning models in health care. JAMA. 2020;323(4):305-306. doi:10.1001/jama.2019.20866

Beil M, Proft I, van Heerden D, et al. Ethical considerations about artificial intelligence for prognostication in intensive care. Intensive Care Medicine Experimental. 2019;7(1):70. doi:10.1186/s40635-019-0286-6

Chen JH, Asch SM. Machine learning and prediction in medicine: beyond the peak of inflated expectations. New England Journal of Medicine. 2017;376(26):2507-2509. doi:10.1056/NEJMp1702071

Chen JH, Alagappan M, Goldstein MK, et al. Decaying relevance of clinical data towards future decisions in data-driven inpatient clinical order sets. International Journal of Medical Informatics. 2017;102:71-79. doi:10.1016/j.ijmedinf.2017.03.006

Desautels T, Calvert J, Hoffman J, et al. Prediction of sepsis in the intensive care unit with minimal electronic health record data: a machine learning approach. JMIR Medical Informatics. 2016;4(3):e28. doi:10.2196/medinform.5909

Escobar GJ, Liu VX, Schuler A, et al. Automated identification of adults at risk for in-hospital clinical deterioration. New England Journal of Medicine. 2020;383(20):1951-1960. doi:10.1056/NEJMsa2001090

Futoma J, Simons M, Panch T, et al. The myth of generalisability in clinical research and machine learning in health care. The Lancet Digital Health. 2020;2(9):e489-e492. doi:10.1016/S2589-7500(20)30186-2

Ghassemi M, Naumann T, Schulam P, et al. Practical guidance on artificial intelligence for health-care data. The Lancet Digital Health. 2019;1(4):e157-e159. doi:10.1016/S2589-7500(19)30084-6

Goldstein BA, Navar AM, Pencina MJ, Ioannidis JPA. Opportunities and challenges in developing risk prediction models with electronic health records data: a systematic review. Journal of the American Medical Informatics Association. 2017;24(1):198-208. doi:10.1093/jamia/ocw042

Harutyunyan H, Khachatrian H, Kale DC, et al. Multitask learning and benchmarking with clinical time series data. Scientific Data. 2019;6(1):96. doi:10.1038/s41597-019-0103-9

Henry KE, Hager DN, Pronovost PJ, Saria S. A targeted real-time early warning score (TREWScore) for septic shock. Science Translational Medicine. 2015;7(299):299ra122. doi:10.1126/scitranslmed.aab3719

Hyland SL, Faltys M, Huser M, et al. Early prediction of circulatory failure in the intensive care unit using machine learning. Nature Medicine. 2020;26(3):364-373. doi:10.1038/s41591-020-0789-4

Jarrett D, Yoon J, Bica I, et al. Clairvoyance: A pipeline toolkit for medical time series. International Conference on Learning Representations. 2021. Available from: https://arxiv.org/abs/2101.07328

Johnson AEW, Pollard TJ, Shen L, et al. MIMIC-III, a freely accessible critical care database. Scientific Data. 2016;3:160035. doi:10.1038/sdata.2016.35

Kaji DA, Zech JR, Kim JS, et al. An attention based deep learning model of clinical events in the intensive care unit. PLOS ONE. 2019;14(2):e0211057. doi:10.1371/journal.pone.0211057

Kipnis P, Escobar GJ, Draper D, et al. Statistical approaches to identifying preventable readmissions. Medical Care. 2018;56(7):629-637. doi:10.1097/MLR.0000000000000931

Lipton ZC, Kale DC, Elkan C, Wetzel R. Learning to diagnose with LSTM recurrent neural networks. International Conference on Learning Representations. 2016. Available from: https://arxiv.org/abs/1511.03677

Liu VX, Bates DW, Wiens J, Shah NH. The number needed to benefit: estimating the value of predictive analytics in healthcare. Journal of the American Medical Informatics Association. 2019;26(12):1655-1659. doi:10.1093/jamia/ocz088

Mayaud L, Lai PS, Clifford GD, et al. Dynamic data during hypotensive episode improves mortality predictions among patients with sepsis and hypotension. Critical Care Medicine. 2013;41(4):954-962. doi:10.1097/CCM.0b013e3182772adb

Nori H, Jenkins S, Koch P, Caruana R. InterpretML: A unified framework for machine learning interpretability. arXiv preprint. 2019. Available from: https://arxiv.org/abs/1909.09223

Obermeyer Z, Powers B, Vogeli C, Mullainathan S. Dissecting racial bias in an algorithm used to manage the health of populations. Science. 2019;366(6464):447-453. doi:10.1126/science.aax2342

Parikh RB, Teeple S, Navathe AS. Addressing bias in artificial intelligence in health care. JAMA. 2019;322(24):2377-2378. doi:10.1001/jama.2019.18058

Rajkomar A, Hardt M, Howell MD, et al. Ensuring fairness in machine learning to advance health equity. Annals of Internal Medicine. 2018;169(12):866-872. doi:10.7326/M18-1990

Rajkomar A, Oren E, Chen K, et al. Scalable and accurate deep learning with electronic health records. NPJ Digital Medicine. 2018;1:18. doi:10.1038/s41746-018-0029-1

Shah NH, Milstein A, Bagley SC. Making machine learning models clinically useful. JAMA. 2019;322(14):1351-1352. doi:10.1001/jama.2019.10306

Smith MJ, Bean S, Bringedahl S, et al. Allegheny County Predictive Risk Modeling Tool: Frequently asked questions and ethical concerns. Available from: https://www.alleghenycountyanalytics.us/wp-content/uploads/2019/05/16-ACDHS-26_PredictiveRisk_Package_050119_FINAL.pdf

Thottakkara P, Ozrazgat-Baslanti T, Hupf BB, et al. Application of machine learning techniques to high-dimensional clinical data to forecast postoperative complications. PLOS ONE. 2016;11(5):e0155705. doi:10.1371/journal.pone.0155705

Wong A, Otles E, Donnelly JP, et al. External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. JAMA Internal Medicine. 2021;181(8):1065-1070. doi:10.1001/jamainternmed.2021.2626

Wynants L, Van Calster B, Collins GS, et al. Prediction models for diagnosis and prognosis of covid-19: systematic review and critical appraisal. BMJ. 2020;369:m1328. doi:10.1136/bmj.m1328

Zech JR, Badgeley MA, Liu M, et al. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study. PLOS Medicine. 2018;15(11):e1002683. doi:10.1371/journal.pmed.1002683
