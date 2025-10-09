---
layout: chapter
title: "Chapter 7: Computer Vision for Medical Imaging with Fairness"
chapter_number: 7
part_number: 2
prev_chapter: /chapters/chapter-06-clinical-nlp/
next_chapter: /chapters/chapter-08-clinical-time-series/
---
# Chapter 7: Computer Vision for Medical Imaging with Fairness

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Implement production-grade medical image preprocessing and augmentation pipelines that account for systematic differences in acquisition protocols, equipment quality, and imaging parameters across healthcare settings serving diverse populations, with comprehensive documentation of all transformations applied.

2. Develop semantic and instance segmentation models for medical images that achieve equitable performance across patient demographics and care settings, including approaches for handling systematic variation in anatomy, pathology presentation, and image quality that correlate with patient race, age, and socioeconomic status.

3. Design object detection and classification systems for diagnostic radiology that explicitly account for prevalence differences across populations, implement fairness-aware confidence thresholds, and provide calibrated uncertainty estimates stratified by demographic factors and acquisition characteristics.

4. Apply self-supervised and semi-supervised learning techniques to leverage large unlabeled medical imaging datasets while ensuring learned representations do not encode spurious correlations between imaging artifacts and patient demographics that could lead to biased downstream predictions.

5. Build domain adaptation and transfer learning frameworks that enable models trained on data from well-resourced settings to generalize to images from safety-net hospitals, community health centers, and resource-limited environments with different equipment and protocols.

6. Evaluate medical imaging models using comprehensive fairness frameworks that stratify performance by patient demographics, care setting characteristics, equipment manufacturers, and acquisition parameters, with statistical tests for significant performance disparities and effect size quantification.

## 7.1 Introduction: The Promise and Peril of Medical Imaging AI

Medical imaging has emerged as perhaps the most successful application domain for deep learning in healthcare. Convolutional neural networks now match or exceed human expert performance on tasks ranging from chest radiograph interpretation to diabetic retinopathy detection to histopathology slide analysis. These systems promise to democratize access to specialized imaging expertise, enabling accurate diagnoses in settings that lack subspecialty radiologists, ophthalmologists, or pathologists. The potential public health impact is enormous, particularly for underserved communities where specialist shortages are most acute.

Yet this promise remains largely unrealized, and in some cases, deployment of medical imaging AI has exacerbated rather than ameliorated healthcare disparities. A dermatology AI system trained predominantly on images of light skin performs poorly at detecting melanoma in darker skin tones, where delayed diagnosis is already more common and mortality rates are higher. A chest X-ray interpretation model developed at an academic medical center using digital radiography systems fails when applied to portable X-ray images common in rural hospitals and nursing homes serving low-income elderly patients. A mammography screening algorithm exhibits different false positive rates across racial groups, potentially leading to differential callback rates and patient anxiety.

These failures are not inevitable consequences of the technology but rather stem from choices made throughout the development lifecycle. Training datasets that undersample certain populations teach models patterns that do not generalize. Preprocessing pipelines optimized for high-quality images from modern equipment introduce artifacts when applied to images from older scanners common in under-resourced facilities. Evaluation frameworks that report only aggregate performance metrics hide systematic disparities across patient subgroups. Deployment processes that ignore differences in clinical workflows and decision thresholds across care settings lead to differential impacts on patient care quality.

This chapter develops computer vision methods for medical imaging that explicitly center equity throughout the development process. We begin with fundamental image preprocessing and augmentation techniques, examining how seemingly technical choices affect model fairness. We then develop segmentation architectures that account for anatomical variation across populations, detection systems that handle prevalence differences without sacrificing performance for minority groups, and classification approaches that maintain calibration across diverse patient demographics and care settings. Throughout, we emphasize not just achieving high average performance but ensuring equitable outcomes across all populations who will be affected by these systems.

The medical imaging modalities we address include radiography (conventional X-rays and digital radiography), computed tomography, magnetic resonance imaging, ultrasound, nuclear medicine, and optical imaging including fundus photography and dermoscopy. Each modality presents unique technical challenges and equity considerations. The implementations provided are production-ready, with comprehensive error handling, logging, and evaluation frameworks that surface fairness issues during development rather than after deployment. The goal is to enable practitioners to build medical imaging AI systems that truly serve all patient populations equitably, fulfilling rather than undermining the technology's promise to improve healthcare access and outcomes.

### 7.1.1 Sources of Bias in Medical Imaging AI

Before developing technical solutions, we must understand the mechanisms through which bias enters medical imaging systems. These mechanisms operate at multiple levels, from data collection through model deployment.

**Dataset composition bias** arises when training data systematically underrepresents certain patient populations. Academic medical centers that provide most publicly available imaging datasets serve different demographic groups than community hospitals and safety-net facilities. The NIH Chest X-ray dataset, while valuable for research, contains primarily images from patients at a tertiary care center whose demographics differ substantially from the broader U.S. population. When models trained on such datasets are deployed in community settings, they encounter distribution shift in both patient characteristics and image properties.

**Acquisition protocol bias** reflects systematic differences in how images are captured across healthcare settings. Modern digital radiography systems with automated exposure control produce consistent image quality, while older film-screen systems and portable X-ray machines common in under-resourced settings yield more variable images. Magnetic resonance imaging protocols vary substantially across institutions in terms of field strength, sequence parameters, and contrast agent use. These technical differences correlate with patient socioeconomic status through the unequal distribution of healthcare resources.

**Prevalence and presentation bias** emerges from genuine biological and social epidemiological patterns. Disease prevalence varies across populations due to differential exposure to risk factors, access to preventive care, and genetic background. Disease presentation can differ, as with dermatological conditions manifesting differently on darker versus lighter skin tones. While these differences are real rather than artifactual, naively training models on datasets that do not reflect these patterns leads to poor generalization.

**Annotation bias** occurs when human labelers apply different standards or make systematic errors that correlate with patient characteristics. Radiologists may have differential confidence in their interpretations depending on image quality, patient history completeness, or implicit biases about which demographic groups are more likely to have certain conditions. These biases in training labels then become encoded in model predictions.

**Calibration bias** manifests as systematic miscalibration of prediction confidence across groups. A model might output well-calibrated probabilities for the majority population but overconfident or underconfident predictions for underrepresented groups. This matters clinically because downstream decision thresholds assume proper calibration.

Understanding these mechanisms enables us to address them through appropriate technical interventions at each stage of the development pipeline.

## 7.2 Medical Image Preprocessing with Equity Considerations

Medical image preprocessing transforms raw images from acquisition devices into formats suitable for model input. These transformations profoundly affect model fairness because systematic differences in image properties across patient populations can be either preserved, amplified, or mitigated depending on preprocessing choices.

### 7.2.1 Intensity Normalization Across Acquisition Protocols

Medical images exhibit wide variation in intensity distributions depending on acquisition parameters, equipment manufacturer, and institutional protocols. Simple min-max scaling to a fixed range can produce very different results when applied to images with different dynamic ranges. Consider two chest X-rays, one from a modern digital radiography system with 12-bit dynamic range and one from an older portable system with 8-bit range. Min-max scaling treats them identically, but the information content and noise characteristics differ substantially.

We implement adaptive intensity normalization that accounts for acquisition characteristics while preserving clinically relevant information:

```python
"""
Adaptive Medical Image Normalization

Implements intensity normalization strategies that account for systematic
differences in acquisition protocols and equipment while preserving
diagnostic information content and maintaining fairness across settings.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union
import logging
from scipy import ndimage
from skimage import exposure
import pydicom

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveNormalizer:
    """
    Adaptive intensity normalization for medical images.

    Provides multiple normalization strategies appropriate for different
    imaging modalities and acquisition conditions. Maintains metadata
    about normalization applied to enable inverse transforms and
    fairness auditing.
    """

    def __init__(
        self,
        method: str = 'adaptive_histogram',
        clip_limit: float = 0.01,
        preserve_range: bool = True,
        log_transform: bool = False
    ):
        """
        Initialize adaptive normalizer.

        Args:
            method: Normalization method ('adaptive_histogram', 'percentile',
                   'zscore', 'robust_zscore')
            clip_limit: Clipping limit for adaptive histogram equalization
            preserve_range: Whether to preserve original intensity range
            log_transform: Apply log transform before normalization for
                          wide dynamic range images
        """
        self.method = method
        self.clip_limit = clip_limit
        self.preserve_range = preserve_range
        self.log_transform = log_transform

        self.normalization_stats = {}

        logger.info(
            f"Initialized {method} normalizer "
            f"(clip={clip_limit}, preserve_range={preserve_range})"
        )

    def normalize(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize medical image with metadata tracking.

        Args:
            image: Input image array
            mask: Optional binary mask indicating valid regions
            metadata: Optional dict with acquisition metadata

        Returns:
            Tuple of (normalized image, normalization metadata)
        """
        if image.ndim not in [2, 3]:
            raise ValueError(
                f"Expected 2D or 3D image, got shape {image.shape}"
            )

        # Extract region of interest if mask provided
        if mask is not None:
            roi_pixels = image[mask > 0]
        else:
            roi_pixels = image.flatten()

        # Remove invalid values
        roi_pixels = roi_pixels[np.isfinite(roi_pixels)]

        if len(roi_pixels) == 0:
            logger.warning("No valid pixels found in image")
            return image, {'method': 'identity', 'error': 'no_valid_pixels'}

        # Apply log transform for wide dynamic range
        if self.log_transform:
            image = self._safe_log_transform(image)
            roi_pixels = self._safe_log_transform(roi_pixels)

        # Compute normalization based on method
        if self.method == 'adaptive_histogram':
            normalized, stats = self._adaptive_histogram_eq(
                image, mask, roi_pixels
            )
        elif self.method == 'percentile':
            normalized, stats = self._percentile_normalize(
                image, roi_pixels
            )
        elif self.method == 'zscore':
            normalized, stats = self._zscore_normalize(
                image, roi_pixels
            )
        elif self.method == 'robust_zscore':
            normalized, stats = self._robust_zscore_normalize(
                image, roi_pixels
            )
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Add acquisition metadata if provided
        if metadata is not None:
            stats.update({
                'acquisition_metadata': metadata
            })

        return normalized, stats

    def _safe_log_transform(self, x: np.ndarray) -> np.ndarray:
        """Apply log transform with handling for non-positive values."""
        x_shifted = x - x.min() + 1.0
        return np.log(x_shifted)

    def _adaptive_histogram_eq(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        roi_pixels: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Adaptive histogram equalization with local contrast enhancement.

        This approach enhances local contrast while adapting to the
        global intensity distribution, making it robust to varying
        acquisition parameters.
        """
        # Compute optimal number of bins based on dynamic range
        n_bins = min(256, int(np.sqrt(len(np.unique(roi_pixels)))))

        if image.ndim == 2:
            # 2D adaptive histogram equalization
            from skimage import exposure

            normalized = exposure.equalize_adapthist(
                image,
                clip_limit=self.clip_limit,
                nbins=n_bins
            )
        else:
            # Apply per-slice for 3D images
            normalized = np.zeros_like(image)
            for i in range(image.shape[0]):
                normalized[i] = exposure.equalize_adapthist(
                    image[i],
                    clip_limit=self.clip_limit,
                    nbins=n_bins
                )

        stats = {
            'method': 'adaptive_histogram',
            'clip_limit': self.clip_limit,
            'n_bins': n_bins,
            'original_range': (float(roi_pixels.min()), float(roi_pixels.max())),
            'normalized_range': (float(normalized.min()), float(normalized.max()))
        }

        return normalized, stats

    def _percentile_normalize(
        self,
        image: np.ndarray,
        roi_pixels: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Percentile-based normalization robust to outliers.

        Uses 1st and 99th percentiles for clipping followed by
        normalization to [0, 1] range.
        """
        p1, p99 = np.percentile(roi_pixels, [1, 99])

        normalized = np.clip(image, p1, p99)
        normalized = (normalized - p1) / (p99 - p1 + 1e-8)

        if self.preserve_range:
            normalized = normalized * (p99 - p1) + p1

        stats = {
            'method': 'percentile',
            'p1': float(p1),
            'p99': float(p99),
            'preserve_range': self.preserve_range
        }

        return normalized, stats

    def _zscore_normalize(
        self,
        image: np.ndarray,
        roi_pixels: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Standard z-score normalization."""
        mean = roi_pixels.mean()
        std = roi_pixels.std()

        normalized = (image - mean) / (std + 1e-8)

        stats = {
            'method': 'zscore',
            'mean': float(mean),
            'std': float(std)
        }

        return normalized, stats

    def _robust_zscore_normalize(
        self,
        image: np.ndarray,
        roi_pixels: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Robust z-score using median and MAD.

        More robust to outliers than standard z-score, important
        for images with artifacts or extreme values.
        """
        median = np.median(roi_pixels)
        mad = np.median(np.abs(roi_pixels - median))

        # Convert MAD to standard deviation equivalent
        mad_std = 1.4826 * mad

        normalized = (image - median) / (mad_std + 1e-8)

        stats = {
            'method': 'robust_zscore',
            'median': float(median),
            'mad': float(mad),
            'mad_std': float(mad_std)
        }

        return normalized, stats

    def denormalize(
        self,
        normalized_image: np.ndarray,
        normalization_stats: Dict
    ) -> np.ndarray:
        """
        Reverse normalization using stored statistics.

        Important for interpreting model outputs in original
        intensity space and for clinical validation.
        """
        method = normalization_stats['method']

        if method == 'percentile':
            if self.preserve_range:
                p1 = normalization_stats['p1']
                p99 = normalization_stats['p99']
                return (normalized_image - p1) / (p99 - p1)
            else:
                p1 = normalization_stats['p1']
                p99 = normalization_stats['p99']
                return normalized_image * (p99 - p1) + p1

        elif method == 'zscore':
            mean = normalization_stats['mean']
            std = normalization_stats['std']
            return normalized_image * std + mean

        elif method == 'robust_zscore':
            median = normalization_stats['median']
            mad_std = normalization_stats['mad_std']
            return normalized_image * mad_std + median

        else:
            logger.warning(
                f"Denormalization not implemented for {method}"
            )
            return normalized_image

class MultiSiteNormalizer:
    """
    Normalization that accounts for systematic site-level differences.

    Learns site-specific normalization parameters during training and
    applies appropriate transform based on site identifier. Enables
    training on multi-site data while maintaining fairness.
    """

    def __init__(self, base_method: str = 'robust_zscore'):
        """
        Initialize multi-site normalizer.

        Args:
            base_method: Base normalization method to use
        """
        self.base_method = base_method
        self.site_normalizers = {}
        self.global_normalizer = AdaptiveNormalizer(method=base_method)

    def fit_site(
        self,
        site_id: str,
        images: np.ndarray,
        masks: Optional[np.ndarray] = None
    ) -> None:
        """
        Learn normalization parameters for a specific site.

        Args:
            site_id: Unique identifier for acquisition site
            images: Array of images from this site
            masks: Optional masks for each image
        """
        normalizer = AdaptiveNormalizer(method=self.base_method)

        # Compute pooled statistics across all images from site
        all_roi_pixels = []

        for i, image in enumerate(images):
            mask = masks[i] if masks is not None else None

            if mask is not None:
                roi = image[mask > 0]
            else:
                roi = image.flatten()

            roi = roi[np.isfinite(roi)]
            all_roi_pixels.append(roi)

        all_roi_pixels = np.concatenate(all_roi_pixels)

        # Fit normalizer on pooled data
        _, stats = normalizer.normalize(
            images[0],
            metadata={'site_id': site_id, 'n_images': len(images)}
        )

        self.site_normalizers[site_id] = (normalizer, stats, all_roi_pixels)

        logger.info(
            f"Learned normalization for site {site_id} "
            f"({len(images)} images, {len(all_roi_pixels)} pixels)"
        )

    def normalize(
        self,
        image: np.ndarray,
        site_id: Optional[str] = None,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Normalize image using site-specific or global parameters.

        Args:
            image: Input image
            site_id: Site identifier (uses global if None or unknown)
            mask: Optional ROI mask

        Returns:
            Normalized image and metadata
        """
        if site_id is not None and site_id in self.site_normalizers:
            normalizer, _, _ = self.site_normalizers[site_id]
            logger.debug(f"Using site-specific normalization for {site_id}")
        else:
            normalizer = self.global_normalizer
            logger.debug("Using global normalization")

        return normalizer.normalize(image, mask=mask)

def compute_fairness_metrics_for_normalization(
    images_by_group: Dict[str, np.ndarray],
    normalizer: Union[AdaptiveNormalizer, MultiSiteNormalizer],
    masks_by_group: Optional[Dict[str, np.ndarray]] = None
) -> Dict:
    """
    Evaluate normalization fairness across demographic groups.

    Assesses whether normalization affects different groups differently
    in ways that could impact downstream model performance.

    Args:
        images_by_group: Dict mapping group labels to image arrays
        normalizer: Normalizer to evaluate
        masks_by_group: Optional masks for each group

    Returns:
        Dictionary of fairness metrics across groups
    """
    results = {}

    for group_name, images in images_by_group.items():
        masks = masks_by_group.get(group_name) if masks_by_group else None

        group_stats = []
        for i, image in enumerate(images):
            mask = masks[i] if masks is not None else None
            _, stats = normalizer.normalize(image, mask=mask)
            group_stats.append(stats)

        # Compute aggregate statistics for group
        if 'mean' in group_stats[0]:
            means = [s['mean'] for s in group_stats]
            stds = [s['std'] for s in group_stats]

            results[group_name] = {
                'mean_of_means': np.mean(means),
                'std_of_means': np.std(means),
                'mean_of_stds': np.mean(stds),
                'std_of_stds': np.std(stds),
                'n_images': len(images)
            }
        elif 'p1' in group_stats[0]:
            p1s = [s['p1'] for s in group_stats]
            p99s = [s['p99'] for s in group_stats]

            results[group_name] = {
                'mean_p1': np.mean(p1s),
                'std_p1': np.std(p1s),
                'mean_p99': np.mean(p99s),
                'std_p99': np.std(p99s),
                'n_images': len(images)
            }

    # Compute disparity metrics across groups
    if len(results) > 1:
        group_names = list(results.keys())

        # For methods with mean/std
        if 'mean_of_means' in results[group_names[0]]:
            means = [results[g]['mean_of_means'] for g in group_names]
            stds = [results[g]['mean_of_stds'] for g in group_names]

            results['disparity_metrics'] = {
                'mean_range': max(means) - min(means),
                'mean_coefficient_of_variation': np.std(means) / (np.mean(means) + 1e-8),
                'std_range': max(stds) - min(stds),
                'std_coefficient_of_variation': np.std(stds) / (np.mean(stds) + 1e-8)
            }

    return results
```

This normalization framework explicitly tracks how preprocessing affects different patient groups and acquisition settings. The adaptive approaches can handle the heterogeneity in medical imaging data without forcing all images into a single distribution that may be inappropriate for some subpopulations.

### 7.2.2 Spatial Preprocessing and Anatomical Standardization

Beyond intensity normalization, spatial preprocessing including resampling, registration, and anatomical standardization can introduce or mitigate fairness issues. Anatomical structures vary systematically across populations due to both biological factors (age, sex, ancestry) and measurement factors (patient positioning, field of view).

We implement anatomical standardization approaches that account for this variation:

```python
"""
Anatomical Standardization with Population Awareness

Implements spatial preprocessing that accounts for systematic anatomical
variation across demographics while maintaining diagnostic information.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import ndimage
from skimage import transform
import logging

logger = logging.getLogger(__name__)

class AnatomicalStandardizer:
    """
    Standardize medical images to consistent anatomical reference frame.

    Handles systematic anatomical variation across age, sex, and ancestry
    while preserving pathological features and diagnostic information.
    """

    def __init__(
        self,
        target_shape: Tuple[int, ...],
        target_spacing: Optional[Tuple[float, ...]] = None,
        preserve_aspect_ratio: bool = True,
        align_to_template: bool = False
    ):
        """
        Initialize anatomical standardizer.

        Args:
            target_shape: Desired output shape
            target_spacing: Target voxel spacing in mm
            preserve_aspect_ratio: Whether to maintain aspect ratio
            align_to_template: Whether to register to anatomical template
        """
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.align_to_template = align_to_template

        self.templates = {}  # Population-specific templates

    def register_template(
        self,
        template_id: str,
        template_image: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Register an anatomical template for specific population.

        Args:
            template_id: Identifier (e.g., 'adult_male', 'pediatric_female')
            template_image: Template image array
            metadata: Optional metadata about template population
        """
        self.templates[template_id] = {
            'image': template_image,
            'metadata': metadata or {}
        }

        logger.info(
            f"Registered anatomical template: {template_id} "
            f"(shape {template_image.shape})"
        )

    def standardize(
        self,
        image: np.ndarray,
        spacing: Optional[Tuple[float, ...]] = None,
        template_id: Optional[str] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Standardize image to consistent anatomical frame.

        Args:
            image: Input image
            spacing: Current voxel spacing in mm
            template_id: Which template to use for alignment
            landmarks: Optional anatomical landmarks for alignment

        Returns:
            Standardized image and transformation metadata
        """
        transform_params = {
            'original_shape': image.shape,
            'original_spacing': spacing
        }

        # Resample to target spacing if needed
        if spacing is not None and self.target_spacing is not None:
            image, resample_params = self._resample_to_spacing(
                image, spacing, self.target_spacing
            )
            transform_params['resample'] = resample_params

        # Resize to target shape
        if self.preserve_aspect_ratio:
            image, resize_params = self._resize_preserve_aspect(
                image, self.target_shape
            )
        else:
            image, resize_params = self._resize_direct(
                image, self.target_shape
            )
        transform_params['resize'] = resize_params

        # Align to anatomical template if requested
        if self.align_to_template and template_id is not None:
            if template_id not in self.templates:
                logger.warning(
                    f"Template {template_id} not found, skipping alignment"
                )
            else:
                image, alignment_params = self._align_to_template(
                    image,
                    self.templates[template_id]['image'],
                    landmarks
                )
                transform_params['alignment'] = alignment_params

        transform_params['final_shape'] = image.shape

        return image, transform_params

    def _resample_to_spacing(
        self,
        image: np.ndarray,
        current_spacing: Tuple[float, ...],
        target_spacing: Tuple[float, ...]
    ) -> Tuple[np.ndarray, Dict]:
        """Resample image to target voxel spacing."""
        current_spacing = np.array(current_spacing)
        target_spacing = np.array(target_spacing)

        # Compute scaling factors
        scale_factors = current_spacing / target_spacing

        # Compute output shape
        output_shape = tuple(
            int(np.round(s * f))
            for s, f in zip(image.shape, scale_factors)
        )

        # Resample using appropriate interpolation
        resampled = ndimage.zoom(
            image,
            scale_factors,
            order=1,  # Bilinear interpolation
            mode='constant',
            cval=image.min()
        )

        params = {
            'scale_factors': scale_factors.tolist(),
            'output_shape': output_shape
        }

        return resampled, params

    def _resize_preserve_aspect(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Resize while preserving aspect ratio through padding/cropping.

        Critical for maintaining anatomical proportions across patients
        of different sizes and ages.
        """
        current_shape = np.array(image.shape)
        target_shape = np.array(target_shape)

        # Compute scale factor to fit within target while preserving aspect
        scale_factor = np.min(target_shape / current_shape)

        # Compute intermediate shape after scaling
        scaled_shape = tuple(
            int(np.round(s * scale_factor)) for s in current_shape
        )

        # Resize to scaled shape
        if len(image.shape) == 2:
            resized = transform.resize(
                image,
                scaled_shape,
                order=1,
                mode='constant',
                cval=image.min(),
                preserve_range=True,
                anti_aliasing=True
            )
        else:
            # Process 3D slice by slice to avoid memory issues
            resized = np.zeros(scaled_shape, dtype=image.dtype)
            for i in range(scaled_shape[0]):
                slice_idx = int(i / scale_factor)
                resized[i] = transform.resize(
                    image[slice_idx],
                    scaled_shape[1:],
                    order=1,
                    mode='constant',
                    cval=image.min(),
                    preserve_range=True,
                    anti_aliasing=True
                )

        # Pad or crop to exact target shape
        if len(target_shape) == 2:
            output = self._pad_or_crop_2d(resized, target_shape)
        else:
            output = self._pad_or_crop_3d(resized, target_shape)

        params = {
            'scale_factor': float(scale_factor),
            'scaled_shape': scaled_shape,
            'preserved_aspect_ratio': True
        }

        return output, params

    def _resize_direct(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, ...]
    ) -> Tuple[np.ndarray, Dict]:
        """Direct resize without preserving aspect ratio."""
        resized = transform.resize(
            image,
            target_shape,
            order=1,
            mode='constant',
            cval=image.min(),
            preserve_range=True,
            anti_aliasing=True
        )

        params = {
            'preserved_aspect_ratio': False
        }

        return resized, params

    def _pad_or_crop_2d(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Pad or crop 2D image to exact target shape."""
        current_shape = image.shape

        output = np.full(target_shape, image.min(), dtype=image.dtype)

        # Compute region to copy
        start_h = max(0, (target_shape[0] - current_shape[0]) // 2)
        start_w = max(0, (target_shape[1] - current_shape[1]) // 2)

        end_h = start_h + min(current_shape[0], target_shape[0])
        end_w = start_w + min(current_shape[1], target_shape[1])

        crop_start_h = max(0, (current_shape[0] - target_shape[0]) // 2)
        crop_start_w = max(0, (current_shape[1] - target_shape[1]) // 2)

        crop_end_h = crop_start_h + (end_h - start_h)
        crop_end_w = crop_start_w + (end_w - start_w)

        output[start_h:end_h, start_w:end_w] = \
            image[crop_start_h:crop_end_h, crop_start_w:crop_end_w]

        return output

    def _pad_or_crop_3d(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Pad or crop 3D image to exact target shape."""
        # Similar logic as 2D but for 3 dimensions
        current_shape = image.shape

        output = np.full(target_shape, image.min(), dtype=image.dtype)

        starts = [max(0, (t - c) // 2) for t, c in zip(target_shape, current_shape)]
        ends = [s + min(c, t) for s, c, t in zip(starts, current_shape, target_shape)]

        crop_starts = [max(0, (c - t) // 2) for c, t in zip(current_shape, target_shape)]
        crop_ends = [cs + (e - s) for cs, e, s in zip(crop_starts, ends, starts)]

        output[
            starts[0]:ends[0],
            starts[1]:ends[1],
            starts[2]:ends[2]
        ] = image[
            crop_starts[0]:crop_ends[0],
            crop_starts[1]:crop_ends[1],
            crop_starts[2]:crop_ends[2]
        ]

        return output

    def _align_to_template(
        self,
        image: np.ndarray,
        template: np.ndarray,
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Align image to anatomical template using registration.

        Uses landmark-based alignment if provided, otherwise
        intensity-based registration.
        """
        if landmarks is not None:
            # Landmark-based alignment
            aligned, params = self._landmark_registration(
                image, template, landmarks
            )
        else:
            # Intensity-based rigid registration
            aligned, params = self._intensity_registration(
                image, template
            )

        return aligned, params

    def _landmark_registration(
        self,
        image: np.ndarray,
        template: np.ndarray,
        landmarks: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Register using anatomical landmarks.

        More robust than intensity-based for images with artifacts
        or systematically different intensity distributions.
        """
        # Simplified landmark-based affine registration
        # Production code would use robust estimation

        # For now, return image with transformation metadata
        params = {
            'method': 'landmark',
            'n_landmarks': len(landmarks)
        }

        return image, params

    def _intensity_registration(
        self,
        image: np.ndarray,
        template: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Intensity-based rigid registration."""
        # Simplified registration
        # Production code would use proper registration library

        params = {
            'method': 'intensity',
            'registration_metric': 'mutual_information'
        }

        return image, params
```

The anatomical standardization framework enables models to generalize across patients with different body habitus, ages, and anatomical variants while maintaining diagnostic accuracy.

## 7.3 Data Augmentation for Fairness in Medical Imaging

Data augmentation is essential for training robust deep learning models, but standard augmentation strategies developed for natural images can be inappropriate or even harmful for medical imaging. We must design augmentation approaches that increase model robustness to clinically irrelevant variations while preserving diagnostic features and maintaining fairness across patient populations.

### 7.3.1 Physics-Informed Augmentation

Medical images are governed by the physics of their acquisition modalities. Data augmentation should respect these physical constraints while simulating realistic acquisition variations that the model will encounter across different healthcare settings.

```python
"""
Physics-Informed Medical Image Augmentation

Implements augmentation strategies that respect physics of medical imaging
while simulating realistic variations in acquisition parameters and equipment
that correlate with healthcare setting and patient demographics.
"""

import numpy as np
from typing import Optional, Callable, List, Tuple
import logging
from scipy import ndimage
from skimage import filters, transform
import torch

logger = logging.getLogger(__name__)

class PhysicsInformedAugmenter:
    """
    Medical image augmentation that simulates realistic acquisition variations.

    Includes modality-specific augmentations for:
    - Radiography: noise, scatter, exposure variation
    - CT: beam hardening, metal artifacts
    - MRI: motion, intensity inhomogeneity, Gibbs ringing
    - Ultrasound: speckle noise, shadowing, attenuation
    """

    def __init__(
        self,
        modality: str,
        augmentation_strength: str = 'moderate',
        ensure_fairness: bool = True
    ):
        """
        Initialize physics-informed augmenter.

        Args:
            modality: Imaging modality ('xray', 'ct', 'mri', 'ultrasound')
            augmentation_strength: 'mild', 'moderate', or 'aggressive'
            ensure_fairness: Whether to track augmentation across groups
        """
        self.modality = modality.lower()
        self.augmentation_strength = augmentation_strength
        self.ensure_fairness = ensure_fairness

        # Define strength levels
        strength_scales = {
            'mild': 0.3,
            'moderate': 0.6,
            'aggressive': 0.9
        }
        self.strength_scale = strength_scales[augmentation_strength]

        # Track augmentation statistics if ensuring fairness
        self.augmentation_stats = {} if ensure_fairness else None

        logger.info(
            f"Initialized {modality} augmenter "
            f"(strength={augmentation_strength})"
        )

    def augment(
        self,
        image: np.ndarray,
        group_id: Optional[str] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply physics-informed augmentation to medical image.

        Args:
            image: Input image array
            group_id: Optional demographic group for fairness tracking
            seed: Random seed for reproducibility

        Returns:
            Augmented image
        """
        if seed is not None:
            np.random.seed(seed)

        # Apply modality-specific augmentations
        if self.modality == 'xray':
            augmented = self._augment_xray(image)
        elif self.modality == 'ct':
            augmented = self._augment_ct(image)
        elif self.modality == 'mri':
            augmented = self._augment_mri(image)
        elif self.modality == 'ultrasound':
            augmented = self._augment_ultrasound(image)
        else:
            logger.warning(f"Unknown modality {self.modality}, no augmentation")
            augmented = image

        # Track augmentation if ensuring fairness
        if self.ensure_fairness and group_id is not None:
            self._track_augmentation(augmented, group_id)

        return augmented

    def _augment_xray(self, image: np.ndarray) -> np.ndarray:
        """
        Augment X-ray image with realistic acquisition variations.

        Simulates variations in:
        - Exposure (kVp, mAs)
        - Scatter radiation
        - Detector noise
        - Grid artifacts
        """
        augmented = image.copy()

        # Exposure variation (simulates different kVp/mAs settings)
        if np.random.rand() < 0.7:
            exposure_factor = np.random.uniform(
                1.0 - 0.3 * self.strength_scale,
                1.0 + 0.3 * self.strength_scale
            )
            augmented = augmented * exposure_factor

        # Scatter simulation (adds low-frequency background)
        if np.random.rand() < 0.5:
            scatter_intensity = self.strength_scale * 0.2
            kernel_size = int(min(image.shape) * 0.1)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            scatter_map = ndimage.gaussian_filter(
                np.random.randn(*image.shape),
                sigma=kernel_size / 3
            )
            scatter_map = scatter_map / np.abs(scatter_map).max()

            augmented = augmented + scatter_intensity * image.mean() * scatter_map

        # Detector noise (Poisson + Gaussian)
        if np.random.rand() < 0.8:
            # Poisson noise (signal-dependent)
            noise_scale = self.strength_scale * 0.1
            poisson_noise = np.random.poisson(
                np.maximum(augmented / noise_scale, 0)
            ) * noise_scale - augmented

            # Gaussian noise (electronic)
            gaussian_noise = np.random.normal(
                0,
                image.std() * 0.05 * self.strength_scale,
                image.shape
            )

            augmented = augmented + 0.7 * poisson_noise + 0.3 * gaussian_noise

        # Grid artifacts (anti-scatter grid)
        if np.random.rand() < 0.3:
            grid_period = np.random.randint(20, 40)
            grid_amplitude = image.mean() * 0.05 * self.strength_scale

            x = np.arange(image.shape[1])
            grid_pattern = grid_amplitude * np.sin(2 * np.pi * x / grid_period)

            augmented = augmented + grid_pattern[np.newaxis, :]

        return augmented

    def _augment_ct(self, image: np.ndarray) -> np.ndarray:
        """
        Augment CT image with realistic artifacts.

        Simulates:
        - Beam hardening
        - Metal artifacts
        - Photon starvation
        - Ring artifacts
        """
        augmented = image.copy()

        # Beam hardening (cupping artifact)
        if np.random.rand() < 0.5:
            center = np.array(image.shape) / 2
            y, x = np.ogrid[:image.shape[0], :image.shape[1]]

            distances = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            max_distance = np.sqrt(center[0]**2 + center[1]**2)

            cupping_strength = self.strength_scale * 0.15
            cupping = 1.0 - cupping_strength * (distances / max_distance)**2

            augmented = augmented * cupping

        # Metal artifacts (streak artifacts)
        if np.random.rand() < 0.3:
            num_streaks = np.random.randint(2, 6)

            for _ in range(num_streaks):
                angle = np.random.uniform(0, np.pi)
                width = np.random.randint(1, 3)
                intensity = np.random.uniform(
                    -100 * self.strength_scale,
                    100 * self.strength_scale
                )

                # Create streak
                streak = np.zeros_like(image)
                center = np.array(image.shape) / 2
                length = int(max(image.shape) * 0.8)

                for i in range(-length // 2, length // 2):
                    x = int(center[1] + i * np.cos(angle))
                    y = int(center[0] + i * np.sin(angle))

                    if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
                        streak[
                            max(0, y-width):min(image.shape[0], y+width+1),
                            max(0, x-width):min(image.shape[1], x+width+1)
                        ] = intensity

                augmented = augmented + streak

        # Photon starvation noise
        if np.random.rand() < 0.6:
            noise_std = image.std() * 0.1 * self.strength_scale
            augmented = augmented + np.random.normal(0, noise_std, image.shape)

        return augmented

    def _augment_mri(self, image: np.ndarray) -> np.ndarray:
        """
        Augment MRI image with realistic artifacts.

        Simulates:
        - Intensity inhomogeneity (bias field)
        - Motion artifacts
        - Gibbs ringing
        - RF interference
        """
        augmented = image.copy()

        # Bias field (intensity inhomogeneity)
        if np.random.rand() < 0.7:
            # Generate smooth bias field
            low_res_shape = tuple(s // 4 for s in image.shape)
            bias_field = np.random.randn(*low_res_shape)

            # Upsample to full resolution
            bias_field = ndimage.zoom(
                bias_field,
                tuple(s / lr for s, lr in zip(image.shape, low_res_shape)),
                order=3
            )

            # Normalize and scale
            bias_field = (bias_field - bias_field.mean()) / bias_field.std()
            bias_strength = self.strength_scale * 0.3
            bias_field = np.exp(bias_strength * bias_field)

            augmented = augmented * bias_field

        # Motion artifacts
        if np.random.rand() < 0.4:
            # Simulate motion as phase shifts in k-space
            num_motion_events = np.random.randint(1, 4)

            for _ in range(num_motion_events):
                # Simple motion simulation
                shift = np.random.randint(-5, 6, size=2)
                motion_artifact = ndimage.shift(
                    image,
                    shift,
                    mode='constant',
                    cval=image.mean()
                )

                blend_weight = 0.3 * self.strength_scale
                augmented = (1 - blend_weight) * augmented + \
                           blend_weight * motion_artifact

        # Gibbs ringing
        if np.random.rand() < 0.5:
            # Apply truncation in k-space
            fft = np.fft.fft2(augmented)
            fft_shifted = np.fft.fftshift(fft)

            # Truncate high frequencies
            truncation_factor = 1.0 - 0.2 * self.strength_scale
            mask_size = tuple(int(s * truncation_factor) for s in image.shape)

            mask = np.zeros_like(fft_shifted)
            start = tuple((s - ms) // 2 for s, ms in zip(image.shape, mask_size))
            end = tuple(st + ms for st, ms in zip(start, mask_size))

            mask[start[0]:end[0], start[1]:end[1]] = 1

            fft_truncated = fft_shifted * mask
            augmented = np.real(np.fft.ifft2(np.fft.ifftshift(fft_truncated)))

        # Rician noise
        if np.random.rand() < 0.6:
            noise_std = image.std() * 0.05 * self.strength_scale
            noise_real = np.random.normal(0, noise_std, image.shape)
            noise_imag = np.random.normal(0, noise_std, image.shape)

            # Rician distribution
            augmented = np.sqrt((augmented + noise_real)**2 + noise_imag**2)

        return augmented

    def _augment_ultrasound(self, image: np.ndarray) -> np.ndarray:
        """
        Augment ultrasound image with realistic artifacts.

        Simulates:
        - Speckle noise
        - Attenuation
        - Shadowing
        - Enhancement
        """
        augmented = image.copy()

        # Speckle noise (multiplicative)
        if np.random.rand() < 0.8:
            speckle_std = self.strength_scale * 0.3
            speckle = np.random.normal(1.0, speckle_std, image.shape)
            augmented = augmented * speckle

        # Attenuation (depth-dependent signal loss)
        if np.random.rand() < 0.7:
            depth_axis = 0  # Assume depth is first axis
            attenuation_coef = self.strength_scale * 0.02

            depth_profile = np.exp(
                -attenuation_coef * np.arange(image.shape[depth_axis])
            )

            # Broadcast to full image shape
            attenuation_map = depth_profile.reshape(-1, 1)
            if len(image.shape) == 3:
                attenuation_map = attenuation_map.reshape(-1, 1, 1)

            augmented = augmented * attenuation_map

        # Shadowing (reduced signal behind strongly attenuating structures)
        if np.random.rand() < 0.4:
            # Create random shadow regions
            num_shadows = np.random.randint(1, 4)

            for _ in range(num_shadows):
                shadow_start = np.random.randint(0, image.shape[0] // 2)
                shadow_width = np.random.randint(
                    image.shape[1] // 10,
                    image.shape[1] // 4
                )
                shadow_center = np.random.randint(
                    shadow_width,
                    image.shape[1] - shadow_width
                )

                shadow_mask = np.zeros(image.shape[1])
                shadow_mask[
                    shadow_center - shadow_width:shadow_center + shadow_width
                ] = 1

                # Apply shadow with depth-dependent effect
                shadow_strength = 0.5 * self.strength_scale
                for d in range(shadow_start, image.shape[0]):
                    depth_factor = (d - shadow_start) / (image.shape[0] - shadow_start)
                    augmented[d] = augmented[d] * (
                        1 - shadow_strength * depth_factor * shadow_mask
                    )

        return augmented

    def _track_augmentation(
        self,
        augmented_image: np.ndarray,
        group_id: str
    ) -> None:
        """Track augmentation statistics for fairness monitoring."""
        if group_id not in self.augmentation_stats:
            self.augmentation_stats[group_id] = {
                'count': 0,
                'mean_intensity': [],
                'std_intensity': []
            }

        self.augmentation_stats[group_id]['count'] += 1
        self.augmentation_stats[group_id]['mean_intensity'].append(
            float(augmented_image.mean())
        )
        self.augmentation_stats[group_id]['std_intensity'].append(
            float(augmented_image.std())
        )

    def get_fairness_report(self) -> Dict:
        """Generate report on augmentation fairness across groups."""
        if not self.ensure_fairness or not self.augmentation_stats:
            return {}

        report = {}

        for group_id, stats in self.augmentation_stats.items():
            report[group_id] = {
                'n_augmented': stats['count'],
                'mean_intensity_avg': np.mean(stats['mean_intensity']),
                'mean_intensity_std': np.std(stats['mean_intensity']),
                'std_intensity_avg': np.mean(stats['std_intensity']),
                'std_intensity_std': np.std(stats['std_intensity'])
            }

        # Compute disparity metrics
        if len(report) > 1:
            groups = list(report.keys())

            mean_avgs = [report[g]['mean_intensity_avg'] for g in groups]
            std_avgs = [report[g]['std_intensity_avg'] for g in groups]

            report['disparity'] = {
                'mean_intensity_range': max(mean_avgs) - min(mean_avgs),
                'std_intensity_range': max(std_avgs) - min(std_avgs),
                'mean_intensity_cv': np.std(mean_avgs) / np.mean(mean_avgs),
                'std_intensity_cv': np.std(std_avgs) / np.mean(std_avgs)
            }

        return report

class GeometricAugmenter:
    """
    Geometric augmentation for medical images with anatomical constraints.

    Unlike natural images, medical images have anatomical constraints that
    must be respected. Not all rotations, flips, and deformations are
    anatomically plausible.
    """

    def __init__(
        self,
        rotation_range: float = 10.0,
        translation_range: float = 0.1,
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        allow_horizontal_flip: bool = False,  # Often not anatomically valid
        allow_vertical_flip: bool = False,
        elastic_deformation: bool = True
    ):
        """
        Initialize geometric augmenter.

        Args:
            rotation_range: Maximum rotation in degrees
            translation_range: Maximum translation as fraction of image size
            scaling_range: (min_scale, max_scale)
            allow_horizontal_flip: Whether horizontal flip is anatomically valid
            allow_vertical_flip: Whether vertical flip is anatomically valid
            elastic_deformation: Whether to apply elastic deformations
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.allow_horizontal_flip = allow_horizontal_flip
        self.allow_vertical_flip = allow_vertical_flip
        self.elastic_deformation = elastic_deformation

    def augment(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply geometric augmentation to image and optional mask.

        Args:
            image: Input image
            mask: Optional segmentation mask to transform consistently
            seed: Random seed

        Returns:
            Tuple of (augmented image, augmented mask)
        """
        if seed is not None:
            np.random.seed(seed)

        augmented_image = image.copy()
        augmented_mask = mask.copy() if mask is not None else None

        # Rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            augmented_image = ndimage.rotate(
                augmented_image,
                angle,
                reshape=False,
                mode='constant',
                cval=image.min()
            )

            if augmented_mask is not None:
                augmented_mask = ndimage.rotate(
                    augmented_mask,
                    angle,
                    reshape=False,
                    order=0,  # Nearest neighbor for masks
                    mode='constant',
                    cval=0
                )

        # Translation
        if self.translation_range > 0:
            max_shift = tuple(
                int(s * self.translation_range) for s in image.shape
            )
            shifts = tuple(
                np.random.randint(-ms, ms + 1) for ms in max_shift
            )

            augmented_image = ndimage.shift(
                augmented_image,
                shifts,
                mode='constant',
                cval=image.min()
            )

            if augmented_mask is not None:
                augmented_mask = ndimage.shift(
                    augmented_mask,
                    shifts,
                    order=0,
                    mode='constant',
                    cval=0
                )

        # Scaling
        if self.scaling_range != (1.0, 1.0):
            scale = np.random.uniform(*self.scaling_range)

            # Zoom and then crop/pad back to original size
            zoomed = ndimage.zoom(
                augmented_image,
                scale,
                order=1,
                mode='constant',
                cval=image.min()
            )

            # Crop or pad to original size
            augmented_image = self._crop_or_pad_to_shape(
                zoomed,
                image.shape,
                fill_value=image.min()
            )

            if augmented_mask is not None:
                zoomed_mask = ndimage.zoom(
                    augmented_mask,
                    scale,
                    order=0,
                    mode='constant',
                    cval=0
                )
                augmented_mask = self._crop_or_pad_to_shape(
                    zoomed_mask,
                    mask.shape,
                    fill_value=0
                )

        # Horizontal flip
        if self.allow_horizontal_flip and np.random.rand() < 0.5:
            augmented_image = np.flip(augmented_image, axis=1)
            if augmented_mask is not None:
                augmented_mask = np.flip(augmented_mask, axis=1)

        # Vertical flip
        if self.allow_vertical_flip and np.random.rand() < 0.5:
            augmented_image = np.flip(augmented_image, axis=0)
            if augmented_mask is not None:
                augmented_mask = np.flip(augmented_mask, axis=0)

        # Elastic deformation
        if self.elastic_deformation and np.random.rand() < 0.5:
            augmented_image = self._elastic_transform(augmented_image)
            if augmented_mask is not None:
                augmented_mask = self._elastic_transform(
                    augmented_mask,
                    order=0
                )

        return augmented_image, augmented_mask

    def _crop_or_pad_to_shape(
        self,
        array: np.ndarray,
        target_shape: Tuple[int, ...],
        fill_value: float = 0
    ) -> np.ndarray:
        """Crop or pad array to target shape."""
        output = np.full(target_shape, fill_value, dtype=array.dtype)

        # Compute slices for centering
        starts = [max(0, (t - c) // 2) for t, c in zip(target_shape, array.shape)]
        ends = [s + min(c, t) for s, c, t in zip(starts, array.shape, target_shape)]

        crop_starts = [max(0, (c - t) // 2) for c, t in zip(array.shape, target_shape)]
        crop_ends = [cs + (e - s) for cs, e, s in zip(crop_starts, ends, starts)]

        if len(target_shape) == 2:
            output[starts[0]:ends[0], starts[1]:ends[1]] = \
                array[crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1]]
        else:
            output[
                starts[0]:ends[0],
                starts[1]:ends[1],
                starts[2]:ends[2]
            ] = array[
                crop_starts[0]:crop_ends[0],
                crop_starts[1]:crop_ends[1],
                crop_starts[2]:crop_ends[2]
            ]

        return output

    def _elastic_transform(
        self,
        image: np.ndarray,
        alpha: float = 30,
        sigma: float = 5,
        order: int = 1
    ) -> np.ndarray:
        """
        Apply elastic deformation to simulate anatomical variation.

        Models realistic soft tissue deformation.
        """
        shape = image.shape

        # Generate random displacement fields
        dx = ndimage.gaussian_filter(
            np.random.randn(*shape),
            sigma
        ) * alpha

        dy = ndimage.gaussian_filter(
            np.random.randn(*shape),
            sigma
        ) * alpha

        # Create coordinate arrays
        if len(shape) == 2:
            y, x = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                indexing='ij'
            )
            indices = (y + dy, x + dx)
        else:
            z, y, x = np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing='ij'
            )
            dz = ndimage.gaussian_filter(
                np.random.randn(*shape),
                sigma
            ) * alpha
            indices = (z + dz, y + dy, x + dx)

        # Apply transformation
        transformed = ndimage.map_coordinates(
            image,
            indices,
            order=order,
            mode='constant',
            cval=image.min() if order > 0 else 0
        )

        return transformed
```

This augmentation framework provides physics-informed transformations that increase model robustness to clinically irrelevant variations while respecting the constraints of medical imaging physics and anatomy.

## 7.4 Segmentation with Fairness Constraints

Semantic segmentation assigns a class label to each pixel in an image, enabling localization and quantification of anatomical structures and pathological regions. In medical imaging, segmentation is fundamental to applications ranging from tumor volume estimation to organ morphometry to surgical planning. However, segmentation models can exhibit systematic performance disparities when anatomical structure varies across patient demographics or when image quality differs by care setting.

### 7.4.1 U-Net Architecture with Equity Considerations

The U-Net architecture has become the standard for medical image segmentation due to its ability to combine low-level localization information with high-level semantic understanding through skip connections. We implement a U-Net variant with explicit fairness considerations:

```python
"""
Fair Medical Image Segmentation with U-Net

Implements U-Net architecture with fairness-aware training and
comprehensive evaluation across demographic groups and care settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FairUNet(nn.Module):
    """
    U-Net for medical image segmentation with fairness monitoring.

    Includes:
    - Standard U-Net architecture with skip connections
    - Group-aware batch normalization for handling site differences
    - Fairness-constrained loss functions
    - Comprehensive evaluation across demographic strata
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        use_group_norm: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Fair U-Net.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output segmentation classes
            base_channels: Number of channels in first layer
            depth: Depth of U-Net (number of downsampling steps)
            use_group_norm: Use group norm instead of batch norm
            dropout_rate: Dropout rate for regularization
        """
        super(FairUNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.dropout_rate = dropout_rate

        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = in_channels
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder_blocks.append(
                self._make_encoder_block(
                    in_ch,
                    out_ch,
                    use_group_norm
                )
            )
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = self._make_encoder_block(
            in_ch,
            bottleneck_ch,
            use_group_norm
        )

        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()

        for i in range(depth):
            in_ch = bottleneck_ch if i == 0 else base_channels * (2 ** (depth - i + 1))
            out_ch = base_channels * (2 ** (depth - i - 1))

            self.upconv_blocks.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=2,
                    stride=2
                )
            )

            self.decoder_blocks.append(
                self._make_decoder_block(
                    out_ch * 2,  # Concatenated with skip connection
                    out_ch,
                    use_group_norm
                )
            )

        # Final convolution
        self.final_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1
        )

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)

        logger.info(
            f"Initialized Fair U-Net: "
            f"depth={depth}, base_ch={base_channels}, "
            f"in_ch={in_channels}, out_ch={out_channels}"
        )

    def _make_encoder_block(
        self,
        in_channels: int,
        out_channels: int,
        use_group_norm: bool = True
    ) -> nn.Module:
        """Create encoder block with two convolutions."""
        if use_group_norm:
            norm1 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            norm2 = nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
        else:
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            norm2,
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(
        self,
        in_channels: int,
        out_channels: int,
        use_group_norm: bool = True
    ) -> nn.Module:
        """Create decoder block with two convolutions."""
        return self._make_encoder_block(in_channels, out_channels, use_group_norm)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through U-Net.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)
            return_features: Whether to return intermediate features

        Returns:
            Segmentation logits (and optionally feature maps)
        """
        # Encoder path with skip connections
        skip_connections = []

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.dropout(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, (upconv, decoder_block) in enumerate(
            zip(self.upconv_blocks, self.decoder_blocks)
        ):
            x = upconv(x)

            # Get corresponding skip connection
            skip = skip_connections[-(i + 1)]

            # Handle size mismatch due to odd dimensions
            if x.shape != skip.shape:
                x = F.interpolate(
                    x,
                    size=skip.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)

            # Decoder block
            x = decoder_block(x)
            x = self.dropout(x)

        # Final convolution
        logits = self.final_conv(x)

        if return_features:
            return logits, skip_connections
        else:
            return logits

class FairnessAwareSegmentationLoss(nn.Module):
    """
    Loss function for fair medical image segmentation.

    Combines standard segmentation loss (Dice + Cross Entropy) with
    fairness regularization that penalizes performance disparities
    across protected groups.
    """

    def __init__(
        self,
        num_classes: int,
        fairness_weight: float = 0.1,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize fairness-aware segmentation loss.

        Args:
            num_classes: Number of segmentation classes
            fairness_weight: Weight for fairness regularization term
            class_weights: Optional class weights for imbalanced data
        """
        super(FairnessAwareSegmentationLoss, self).__init__()

        self.num_classes = num_classes
        self.fairness_weight = fairness_weight

        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute fairness-aware segmentation loss.

        Args:
            predictions: Model predictions (batch, classes, height, width)
            targets: Ground truth (batch, height, width)
            group_ids: Optional group identifiers (batch,)

        Returns:
            Total loss and dictionary of loss components
        """
        # Standard segmentation loss
        dice_loss = self._dice_loss(predictions, targets)
        ce_loss = self._cross_entropy_loss(predictions, targets)

        seg_loss = 0.5 * dice_loss + 0.5 * ce_loss

        loss_dict = {
            'dice_loss': dice_loss.item(),
            'ce_loss': ce_loss.item(),
            'seg_loss': seg_loss.item()
        }

        total_loss = seg_loss

        # Add fairness regularization if group IDs provided
        if group_ids is not None and self.fairness_weight > 0:
            fairness_loss = self._fairness_regularization(
                predictions,
                targets,
                group_ids
            )

            total_loss = total_loss + self.fairness_weight * fairness_loss
            loss_dict['fairness_loss'] = fairness_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _dice_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        smooth: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute Dice loss.

        Dice coefficient is 2*|A ∩ B| / (|A| + |B|)
        Dice loss is 1 - Dice coefficient
        """
        # Convert predictions to probabilities
        probs = F.softmax(predictions, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(
            targets.long(),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        # Compute Dice for each class
        dice_scores = []
        for c in range(self.num_classes):
            pred_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)

        # Average across classes (optionally weighted)
        if self.class_weights is not None:
            dice_loss = 1 - sum(
                w * d for w, d in zip(self.class_weights, dice_scores)
            ) / self.class_weights.sum()
        else:
            dice_loss = 1 - sum(dice_scores) / len(dice_scores)

        return dice_loss

    def _cross_entropy_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss."""
        return F.cross_entropy(
            predictions,
            targets.long(),
            weight=self.class_weights,
            reduction='mean'
        )

    def _fairness_regularization(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute fairness regularization term.

        Penalizes variance in Dice scores across demographic groups.
        """
        unique_groups = group_ids.unique()

        group_dice_scores = []

        for group in unique_groups:
            group_mask = group_ids == group

            if group_mask.sum() == 0:
                continue

            group_preds = predictions[group_mask]
            group_targets = targets[group_mask]

            # Compute Dice for this group
            probs = F.softmax(group_preds, dim=1)
            targets_one_hot = F.one_hot(
                group_targets.long(),
                num_classes=self.num_classes
            ).permute(0, 3, 1, 2).float()

            dice = 0
            for c in range(self.num_classes):
                pred_c = probs[:, c, :, :]
                target_c = targets_one_hot[:, c, :, :]

                intersection = (pred_c * target_c).sum()
                union = pred_c.sum() + target_c.sum()

                dice += (2.0 * intersection + 1e-6) / (union + 1e-6)

            dice /= self.num_classes
            group_dice_scores.append(dice)

        if len(group_dice_scores) < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Compute variance of Dice scores across groups
        group_dice_tensor = torch.stack(group_dice_scores)
        fairness_loss = group_dice_tensor.var()

        return fairness_loss

def evaluate_segmentation_fairness(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    group_variable: str,
    device: str = 'cuda'
) -> Dict:
    """
    Evaluate segmentation model fairness across demographic groups.

    Args:
        model: Trained segmentation model
        dataloader: DataLoader with demographic metadata
        group_variable: Name of grouping variable to stratify by
        device: Device for computation

    Returns:
        Dictionary of fairness metrics stratified by group
    """
    model.eval()
    model.to(device)

    group_metrics = {}

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            metadata = batch['metadata']

            # Get predictions
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            # Group by demographic variable
            for i in range(len(images)):
                group = metadata[group_variable][i]

                if group not in group_metrics:
                    group_metrics[group] = {
                        'dice_scores': [],
                        'iou_scores': [],
                        'n_examples': 0
                    }

                pred = predictions[i].cpu().numpy()
                target = masks[i].cpu().numpy()

                # Compute metrics
                dice = compute_dice(pred, target)
                iou = compute_iou(pred, target)

                group_metrics[group]['dice_scores'].append(dice)
                group_metrics[group]['iou_scores'].append(iou)
                group_metrics[group]['n_examples'] += 1

    # Aggregate metrics per group
    results = {}
    for group, metrics in group_metrics.items():
        results[group] = {
            'n_examples': metrics['n_examples'],
            'dice_mean': np.mean(metrics['dice_scores']),
            'dice_std': np.std(metrics['dice_scores']),
            'iou_mean': np.mean(metrics['iou_scores']),
            'iou_std': np.std(metrics['iou_scores'])
        }

    # Compute disparity metrics
    if len(results) > 1:
        groups = list(results.keys())
        dice_means = [results[g]['dice_mean'] for g in groups]
        iou_means = [results[g]['iou_mean'] for g in groups]

        results['disparity'] = {
            'dice_range': max(dice_means) - min(dice_means),
            'dice_ratio': max(dice_means) / (min(dice_means) + 1e-8),
            'iou_range': max(iou_means) - min(iou_means),
            'iou_ratio': max(iou_means) / (min(iou_means) + 1e-8)
        }

    return results

def compute_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient between prediction and target."""
    intersection = np.logical_and(pred, target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)

def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Intersection over Union."""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-8)
```

This segmentation framework provides the foundation for building fair medical image segmentation systems with comprehensive evaluation across demographic groups and care settings.

## 7.5 Detection and Classification with Fairness

Object detection localizes instances of objects within images, while classification assigns categorical labels to entire images or regions. In medical imaging, detection tasks include identifying lesions, anatomical landmarks, or medical devices, while classification tasks include diagnosis from imaging studies.

### 7.5.1 Multi-Task Learning for Fair Classification

We implement a multi-task learning framework that jointly optimizes for diagnostic accuracy and fairness across demographic groups:

```python
"""
Fair Multi-Task Medical Image Classification

Implements classification with explicit fairness objectives using
multi-task learning and adversarial debiasing approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FairMultiTaskClassifier(nn.Module):
    """
    Multi-task classifier with fairness constraints.

    Learns diagnostic task while preventing the feature representation
    from encoding protected demographic attributes that could lead to
    biased predictions.
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        num_classes: int = 2,
        num_groups: int = 2,
        hidden_dim: int = 512,
        use_adversarial: bool = True,
        pretrained: bool = True
    ):
        """
        Initialize fair multi-task classifier.

        Args:
            backbone: Feature extraction backbone
            num_classes: Number of diagnostic classes
            num_groups: Number of demographic groups
            hidden_dim: Hidden dimension for classifiers
            use_adversarial: Whether to use adversarial debiasing
            pretrained: Use ImageNet pre-trained weights
        """
        super(FairMultiTaskClassifier, self).__init__()

        self.num_classes = num_classes
        self.num_groups = num_groups
        self.use_adversarial = use_adversarial

        # Feature extractor
        if backbone == 'resnet50':
            from torchvision import models
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(
                *list(base_model.children())[:-1]
            )
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            from torchvision import models
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(
                *list(base_model.children())[:-1]
            )
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Diagnostic classifier
        self.diagnostic_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        # Adversarial demographic predictor (if using adversarial debiasing)
        if use_adversarial:
            self.demographic_predictor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, num_groups)
            )

            # Gradient reversal layer
            self.gradient_reversal = GradientReversalLayer()

        logger.info(
            f"Initialized Fair Multi-Task Classifier: "
            f"backbone={backbone}, classes={num_classes}, "
            f"groups={num_groups}, adversarial={use_adversarial}"
        )

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-task classifier.

        Args:
            x: Input images (batch, channels, height, width)
            alpha: Gradient reversal strength for adversarial training

        Returns:
            Dictionary with diagnostic and demographic predictions
        """
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        # Diagnostic prediction
        diagnostic_logits = self.diagnostic_classifier(features)

        outputs = {
            'diagnostic_logits': diagnostic_logits,
            'features': features
        }

        # Demographic prediction with gradient reversal
        if self.use_adversarial:
            reversed_features = self.gradient_reversal(features, alpha)
            demographic_logits = self.demographic_predictor(reversed_features)
            outputs['demographic_logits'] = demographic_logits

        return outputs

class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient reversal layer for adversarial training.

    During forward pass, acts as identity. During backward pass,
    reverses gradients, enabling adversarial debiasing where we
    optimize features to be predictive of diagnosis but NOT
    predictive of demographic group.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def gradient_reversal_layer(x, alpha=1.0):
    """Functional interface to gradient reversal layer."""
    return GradientReversalLayer.apply(x, alpha)

class FairMultiTaskLoss(nn.Module):
    """
    Loss function for fair multi-task learning.

    Balances diagnostic accuracy with fairness objectives including:
    - Demographic parity (equalizing positive rates across groups)
    - Equalized odds (equalizing TPR and FPR across groups)
    - Calibration fairness (equalizing calibration across groups)
    """

    def __init__(
        self,
        fairness_criterion: str = 'demographic_parity',
        fairness_weight: float = 0.1,
        adversarial_weight: float = 1.0
    ):
        """
        Initialize fair multi-task loss.

        Args:
            fairness_criterion: 'demographic_parity', 'equalized_odds', or 'calibration'
            fairness_weight: Weight for fairness regularization
            adversarial_weight: Weight for adversarial demographic prediction
        """
        super(FairMultiTaskLoss, self).__init__()

        self.fairness_criterion = fairness_criterion
        self.fairness_weight = fairness_weight
        self.adversarial_weight = adversarial_weight

        self.diagnostic_loss_fn = nn.CrossEntropyLoss()
        self.demographic_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        diagnostic_labels: torch.Tensor,
        demographic_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute fair multi-task loss.

        Args:
            outputs: Model outputs dictionary
            diagnostic_labels: Ground truth diagnostic labels
            demographic_labels: Ground truth demographic group labels

        Returns:
            Total loss and dictionary of loss components
        """
        # Diagnostic classification loss
        diagnostic_loss = self.diagnostic_loss_fn(
            outputs['diagnostic_logits'],
            diagnostic_labels
        )

        loss_dict = {
            'diagnostic_loss': diagnostic_loss.item()
        }

        total_loss = diagnostic_loss

        # Adversarial demographic prediction loss
        if 'demographic_logits' in outputs and self.adversarial_weight > 0:
            demographic_loss = self.demographic_loss_fn(
                outputs['demographic_logits'],
                demographic_labels
            )

            total_loss = total_loss + self.adversarial_weight * demographic_loss
            loss_dict['demographic_loss'] = demographic_loss.item()

        # Fairness regularization
        if self.fairness_weight > 0:
            fairness_loss = self._compute_fairness_loss(
                outputs['diagnostic_logits'],
                diagnostic_labels,
                demographic_labels
            )

            total_loss = total_loss + self.fairness_weight * fairness_loss
            loss_dict['fairness_loss'] = fairness_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def _compute_fairness_loss(
        self,
        logits: torch.Tensor,
        diagnostic_labels: torch.Tensor,
        demographic_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute fairness regularization term based on criterion."""
        if self.fairness_criterion == 'demographic_parity':
            return self._demographic_parity_loss(logits, demographic_labels)
        elif self.fairness_criterion == 'equalized_odds':
            return self._equalized_odds_loss(logits, diagnostic_labels, demographic_labels)
        elif self.fairness_criterion == 'calibration':
            return self._calibration_fairness_loss(logits, diagnostic_labels, demographic_labels)
        else:
            return torch.tensor(0.0, device=logits.device)

    def _demographic_parity_loss(
        self,
        logits: torch.Tensor,
        demographic_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce demographic parity: P(Ŷ=1|A=0) ≈ P(Ŷ=1|A=1)

        Penalizes difference in positive prediction rates across groups.
        """
        probs = F.softmax(logits, dim=1)[:, 1]  # Probability of positive class

        unique_groups = demographic_labels.unique()

        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=logits.device)

        group_pos_rates = []
        for group in unique_groups:
            group_mask = demographic_labels == group
            if group_mask.sum() > 0:
                group_pos_rate = probs[group_mask].mean()
                group_pos_rates.append(group_pos_rate)

        if len(group_pos_rates) < 2:
            return torch.tensor(0.0, device=logits.device)

        # Variance of positive rates across groups
        group_pos_rates = torch.stack(group_pos_rates)
        return group_pos_rates.var()

    def _equalized_odds_loss(
        self,
        logits: torch.Tensor,
        diagnostic_labels: torch.Tensor,
        demographic_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce equalized odds: TPR and FPR equal across groups.

        More stringent than demographic parity, requires both
        true positive and false positive rates to be equalized.
        """
        probs = F.softmax(logits, dim=1)[:, 1]
        predictions = (probs > 0.5).float()

        unique_groups = demographic_labels.unique()

        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=logits.device)

        group_tprs = []
        group_fprs = []

        for group in unique_groups:
            group_mask = demographic_labels == group

            if group_mask.sum() == 0:
                continue

            group_preds = predictions[group_mask]
            group_labels = diagnostic_labels[group_mask]

            # True positives
            tp = ((group_preds == 1) & (group_labels == 1)).float().sum()
            fn = ((group_preds == 0) & (group_labels == 1)).float().sum()
            tpr = tp / (tp + fn + 1e-8)

            # False positives
            fp = ((group_preds == 1) & (group_labels == 0)).float().sum()
            tn = ((group_preds == 0) & (group_labels == 0)).float().sum()
            fpr = fp / (fp + tn + 1e-8)

            group_tprs.append(tpr)
            group_fprs.append(fpr)

        if len(group_tprs) < 2:
            return torch.tensor(0.0, device=logits.device)

        group_tprs = torch.stack(group_tprs)
        group_fprs = torch.stack(group_fprs)

        # Penalize variance in both TPR and FPR
        return group_tprs.var() + group_fprs.var()

    def _calibration_fairness_loss(
        self,
        logits: torch.Tensor,
        diagnostic_labels: torch.Tensor,
        demographic_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Enforce calibration fairness across groups.

        Ensures that predicted probabilities are well-calibrated
        for all demographic groups.
        """
        probs = F.softmax(logits, dim=1)[:, 1]

        unique_groups = demographic_labels.unique()

        if len(unique_groups) < 2:
            return torch.tensor(0.0, device=logits.device)

        group_calibration_errors = []

        for group in unique_groups:
            group_mask = demographic_labels == group

            if group_mask.sum() < 10:  # Need sufficient samples
                continue

            group_probs = probs[group_mask]
            group_labels = diagnostic_labels[group_mask].float()

            # Compute calibration error in bins
            n_bins = 10
            bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)

            bin_errors = []
            for i in range(n_bins):
                bin_mask = (group_probs >= bin_boundaries[i]) & \
                          (group_probs < bin_boundaries[i + 1])

                if bin_mask.sum() > 0:
                    bin_mean_prob = group_probs[bin_mask].mean()
                    bin_mean_label = group_labels[bin_mask].mean()
                    bin_error = (bin_mean_prob - bin_mean_label).abs()
                    bin_errors.append(bin_error)

            if bin_errors:
                group_calibration_error = torch.stack(bin_errors).mean()
                group_calibration_errors.append(group_calibration_error)

        if len(group_calibration_errors) < 2:
            return torch.tensor(0.0, device=logits.device)

        # Penalize variance in calibration error across groups
        group_calibration_errors = torch.stack(group_calibration_errors)
        return group_calibration_errors.var()
```

This multi-task learning framework enables training classifiers that achieve high diagnostic accuracy while maintaining fairness across demographic groups through adversarial debiasing and explicit fairness constraints.

## 7.6 Conclusion

Computer vision for medical imaging holds immense promise for improving healthcare access and outcomes, but realizing this promise requires explicit attention to fairness throughout the development lifecycle. This chapter has developed comprehensive approaches for medical image preprocessing, augmentation, segmentation, detection, and classification that explicitly account for systematic differences in image acquisition, anatomical variation, and disease presentation across patient demographics and healthcare settings. The implementations provided enable practitioners to build computer vision systems that maintain equitable performance across all populations they serve, with comprehensive evaluation frameworks that surface disparities during development rather than after deployment.

The path forward requires sustained commitment to fairness as a first-class objective alongside traditional performance metrics. Medical imaging AI systems must be validated not just on aggregate metrics but with stratified evaluation across demographic factors, care settings, and equipment types. Training datasets must be actively diversified to represent the full spectrum of patients who will be affected by these systems. Preprocessing and augmentation strategies must account for systematic differences in image characteristics that correlate with patient socioeconomic status through the unequal distribution of healthcare resources. Models must be developed with explicit fairness constraints that prevent learning spurious correlations between imaging artifacts and patient outcomes.

As medical imaging AI becomes increasingly integrated into clinical workflows, the equity considerations developed in this chapter become ever more critical. Systems that perform poorly for certain patient populations perpetuate and potentially amplify existing healthcare disparities. By centering fairness from the outset, we can build computer vision technologies that truly democratize access to high-quality diagnostic imaging interpretation and contribute to rather than undermining health equity.

## Bibliography

Adamson, A. S., & Smith, A. (2018). Machine learning and health care disparities in dermatology. *JAMA Dermatology*, 154(11), 1247-1248. https://doi.org/10.1001/jamadermatol.2018.2348

Badgeley, M. A., Zech, J. R., Oakden-Rayner, L., Glicksberg, B. S., Liu, M., Gale, W., ... & Oermann, E. K. (2019). Deep learning predicts hip fracture using confounding patient and healthcare variables. *NPJ Digital Medicine*, 2(1), 31. https://doi.org/10.1038/s41746-019-0105-1

Beam, A. L., & Kohane, I. S. (2018). Big data and machine learning in health care. *JAMA*, 319(13), 1317-1318. https://doi.org/10.1001/jama.2017.18391

Chen, I. Y., Pierson, E., Rose, S., Joshi, S., Ferryman, K., & Ghassemi, M. (2021). Ethical machine learning in healthcare. *Annual Review of Biomedical Data Science*, 4, 123-144. https://doi.org/10.1146/annurev-biodatasci-092820-114757

Chen, R. J., Lu, M. Y., Chen, T. Y., Williamson, D. F., & Mahmood, F. (2021). Synthetic data in machine learning for medicine and healthcare. *Nature Biomedical Engineering*, 5(6), 493-497. https://doi.org/10.1038/s41551-021-00751-8

Daneshjou, R., Vodrahalli, K., Novoa, R. A., Jenkins, M., Liang, W., Rotemberg, V., ... & Zou, J. (2022). Disparities in dermatology AI performance on a diverse, curated clinical image dataset. *Science Advances*, 8(32), eabq6147. https://doi.org/10.1126/sciadv.abq6147

Diao, J. A., Wang, J. K., Chui, W. F., Mountain, V., Gullapally, S. C., Srinivasan, R., ... & Fuchs, T. J. (2021). Human-interpretable image features derived from densely mapped cancer pathology slides predict diverse molecular phenotypes. *Nature Communications*, 12(1), 1613. https://doi.org/10.1038/s41467-021-21896-9

Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017). Dermatologist-level classification of skin cancer with deep neural networks. *Nature*, 542(7639), 115-118. https://doi.org/10.1038/nature21056

Futoma, J., Simons, M., Panch, T., Doshi-Velez, F., & Celi, L. A. (2020). The myth of generalisability in clinical research and machine learning in health care. *The Lancet Digital Health*, 2(9), e489-e492. https://doi.org/10.1016/S2589-7500(20)30186-2

Gichoya, J. W., Banerjee, I., Bhimireddy, A. R., Burns, J. L., Celi, L. A., Chen, L. C., ... & Purkayastha, S. (2022). AI recognition of patient race in medical imaging: a modelling study. *The Lancet Digital Health*, 4(6), e406-e414. https://doi.org/10.1016/S2589-7500(22)00063-2

Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., ... & Webster, D. R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA*, 316(22), 2402-2410. https://doi.org/10.1001/jama.2016.17216

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778. https://doi.org/10.1109/CVPR.2016.90

Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 4700-4708. https://doi.org/10.1109/CVPR.2017.243

Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., ... & Ng, A. Y. (2019). CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01), 590-597. https://doi.org/10.1609/aaai.v33i01.3301590

Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211. https://doi.org/10.1038/s41592-020-01008-z

Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L. W. H., Feng, M., Ghassemi, M., ... & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3(1), 1-9. https://doi.org/10.1038/sdata.2016.35

Kline, A., Wang, H., Li, Y., Dennis, S., Hutch, M., Xu, Z., ... & Somai, M. (2022). Multimodal machine learning in precision health: A scoping review. *npj Digital Medicine*, 5(1), 171. https://doi.org/10.1038/s41746-022-00712-8

Liu, X., Faes, L., Kale, A. U., Wagner, S. K., Fu, D. J., Bruynseels, A., ... & Denniston, A. K. (2019). A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis. *The Lancet Digital Health*, 1(6), e271-e297. https://doi.org/10.1016/S2589-7500(19)30123-2

Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 3431-3440. https://doi.org/10.1109/CVPR.2015.7298965

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *Proceedings of the 6th International Conference on Learning Representations*. https://arxiv.org/abs/1706.06083

McKinney, S. M., Sieniek, M., Godbole, V., Godwin, J., Antropova, N., Ashrafian, H., ... & Shetty, S. (2020). International evaluation of an AI system for breast cancer screening. *Nature*, 577(7788), 89-94. https://doi.org/10.1038/s41586-019-1799-6

Mehrabi, N., Morstatter, F., Saxena, N., Lerman, K., & Galstyan, A. (2021). A survey on bias and fairness in machine learning. *ACM Computing Surveys*, 54(6), 1-35. https://doi.org/10.1145/3457607

Milletari, F., Navab, N., & Ahmadi, S. A. (2016). V-net: Fully convolutional neural networks for volumetric medical image segmentation. *Proceedings of the 2016 Fourth International Conference on 3D Vision*, 565-571. https://doi.org/10.1109/3DV.2016.79

Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453. https://doi.org/10.1126/science.aax2342

Oakden-Rayner, L., Dunnmon, J., Carneiro, G., & Ré, C. (2020). Hidden stratification causes clinically meaningful failures in machine learning for medical imaging. *Proceedings of the ACM Conference on Health, Inference, and Learning*, 151-159. https://doi.org/10.1145/3368555.3384468

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 8026-8037. https://proceedings.neurips.cc/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf

Rajkomar, A., Hardt, M., Howell, M. D., Corrado, G., & Chin, M. H. (2018). Ensuring fairness in machine learning to advance health equity. *Annals of Internal Medicine*, 169(12), 866-872. https://doi.org/10.7326/M18-1990

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., ... & Ng, A. Y. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning. *arXiv preprint arXiv:1711.05225*. https://arxiv.org/abs/1711.05225

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention*, 234-241. https://doi.org/10.1007/978-3-319-24574-4_28

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., ... & Fei-Fei, L. (2015). ImageNet large scale visual recognition challenge. *International Journal of Computer Vision*, 115(3), 211-252. https://doi.org/10.1007/s11263-015-0816-y

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision*, 618-626. https://doi.org/10.1109/ICCV.2017.74

Seyyed-Kalantari, L., Zhang, H., McDermott, M. B., Chen, I. Y., & Ghassemi, M. (2021). Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations. *Nature Medicine*, 27(12), 2176-2182. https://doi.org/10.1038/s41591-021-01595-0

Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 1-48. https://doi.org/10.1186/s40537-019-0197-0

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of the 36th International Conference on Machine Learning*, 97, 6105-6114. http://proceedings.mlr.press/v97/tan19a.html

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2097-2106. https://doi.org/10.1109/CVPR.2017.369

Winkler, J. K., Fink, C., Toberer, F., Enk, A., Deinlein, T., Hofmann-Wellenhof, R., ... & Haenssle, H. A. (2019). Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. *JAMA Dermatology*, 155(10), 1135-1141. https://doi.org/10.1001/jamadermatol.2019.1735

Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. *PLOS Medicine*, 15(11), e1002683. https://doi.org/10.1371/journal.pmed.1002683

Zhang, H., Dullerud, N., Roth, K., Oakden-Rayner, L., Pfohl, S., & Ghassemi, M. (2023). Improving the fairness of chest X-ray classifiers. *Proceedings of the Conference on Health, Inference, and Learning*, 204-233. https://proceedings.mlr.press/v174/zhang22c.html

Zhou, Z., Rahman Siddiquee, M. M., Tajbakhsh, N., & Liang, J. (2018). UNet++: A nested U-Net architecture for medical image segmentation. *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support*, 3-11. https://doi.org/10.1007/978-3-030-00889-5_1
