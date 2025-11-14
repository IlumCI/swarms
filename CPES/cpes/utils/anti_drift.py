"""
Anti-Drift Guards for CPES.

This module implements identity drift prevention and monitoring systems
to ensure the persona maintains consistency over time.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
from loguru import logger
import numpy as np
from pathlib import Path


@dataclass
class DriftMetric:
    """Represents a drift metric measurement."""
    name: str
    value: float
    threshold: float
    timestamp: float
    severity: str  # "low", "medium", "high", "critical"
    description: str


@dataclass
class DriftReport:
    """Represents a drift analysis report."""
    timestamp: float
    total_metrics: int
    violations: int
    critical_violations: int
    metrics: List[DriftMetric]
    recommendations: List[str]
    overall_health: str  # "healthy", "warning", "critical"


class AntiDriftMonitor:
    """
    Anti-drift monitoring system for CPES.
    
    This class monitors various aspects of persona behavior to detect
    and prevent identity drift over time.
    """
    
    def __init__(self, persona, checkpoints_dir: str = "checkpoints"):
        """
        Initialize the anti-drift monitor.
        
        Args:
            persona: Persona object to monitor
            checkpoints_dir: Directory to store drift checkpoints
        """
        self.persona = persona
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        # Drift history
        self.drift_history: List[DriftReport] = []
        self.baseline_metrics: Dict[str, float] = {}
        
        # Monitoring thresholds
        self.thresholds = {
            "sentence_length_variance": 0.3,  # 30% variance allowed
            "tic_usage_frequency": 0.2,  # 20% of responses should have tics
            "value_violation_rate": 0.1,  # 10% violation rate max
            "style_consistency": 0.7,  # 70% style consistency required
            "motive_alignment": 0.8,  # 80% motive alignment required
            "response_length_variance": 0.4,  # 40% length variance allowed
        }
        
        logger.info(f"Initialized AntiDriftMonitor for {persona.spec.name}")
    
    def check_drift(self, recent_responses: List[str], 
                   recent_violations: List[Any] = None) -> DriftReport:
        """
        Check for identity drift in recent responses.
        
        Args:
            recent_responses: List of recent persona responses
            recent_violations: List of recent value violations
            
        Returns:
            DriftReport with analysis results
        """
        logger.info("Checking for identity drift...")
        
        metrics = []
        violations = 0
        critical_violations = 0
        
        # Check sentence length consistency
        sentence_metric = self._check_sentence_length_consistency(recent_responses)
        metrics.append(sentence_metric)
        if sentence_metric.severity in ["high", "critical"]:
            violations += 1
            if sentence_metric.severity == "critical":
                critical_violations += 1
        
        # Check tic usage frequency
        tic_metric = self._check_tic_usage_frequency(recent_responses)
        metrics.append(tic_metric)
        if tic_metric.severity in ["high", "critical"]:
            violations += 1
            if tic_metric.severity == "critical":
                critical_violations += 1
        
        # Check value violation rate
        if recent_violations:
            violation_metric = self._check_value_violation_rate(recent_violations)
            metrics.append(violation_metric)
            if violation_metric.severity in ["high", "critical"]:
                violations += 1
                if violation_metric.severity == "critical":
                    critical_violations += 1
        
        # Check style consistency
        style_metric = self._check_style_consistency(recent_responses)
        metrics.append(style_metric)
        if style_metric.severity in ["high", "critical"]:
            violations += 1
            if style_metric.severity == "critical":
                critical_violations += 1
        
        # Check motive alignment
        motive_metric = self._check_motive_alignment(recent_responses)
        metrics.append(motive_metric)
        if motive_metric.severity in ["high", "critical"]:
            violations += 1
            if motive_metric.severity == "critical":
                critical_violations += 1
        
        # Determine overall health
        if critical_violations > 0:
            overall_health = "critical"
        elif violations > len(metrics) // 2:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, overall_health)
        
        # Create drift report
        report = DriftReport(
            timestamp=time.time(),
            total_metrics=len(metrics),
            violations=violations,
            critical_violations=critical_violations,
            metrics=metrics,
            recommendations=recommendations,
            overall_health=overall_health
        )
        
        # Store report
        self.drift_history.append(report)
        
        logger.info(f"Drift check completed: {overall_health} ({violations} violations)")
        return report
    
    def _check_sentence_length_consistency(self, responses: List[str]) -> DriftMetric:
        """Check consistency of sentence lengths."""
        if not responses:
            return DriftMetric(
                name="sentence_length_variance",
                value=0.0,
                threshold=self.thresholds["sentence_length_variance"],
                timestamp=time.time(),
                severity="low",
                description="No responses to analyze"
            )
        
        # Calculate average sentence lengths
        sentence_lengths = []
        for response in responses:
            sentences = response.split('.')
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths:
                sentence_lengths.extend(lengths)
        
        if not sentence_lengths:
            return DriftMetric(
                name="sentence_length_variance",
                value=0.0,
                threshold=self.thresholds["sentence_length_variance"],
                timestamp=time.time(),
                severity="low",
                description="No sentences found"
            )
        
        # Calculate variance
        mean_length = np.mean(sentence_lengths)
        variance = np.var(sentence_lengths) / (mean_length ** 2) if mean_length > 0 else 0
        
        # Determine severity
        if variance > self.thresholds["sentence_length_variance"] * 2:
            severity = "critical"
        elif variance > self.thresholds["sentence_length_variance"] * 1.5:
            severity = "high"
        elif variance > self.thresholds["sentence_length_variance"]:
            severity = "medium"
        else:
            severity = "low"
        
        return DriftMetric(
            name="sentence_length_variance",
            value=variance,
            threshold=self.thresholds["sentence_length_variance"],
            timestamp=time.time(),
            severity=severity,
            description=f"Sentence length variance: {variance:.3f} (threshold: {self.thresholds['sentence_length_variance']:.3f})"
        )
    
    def _check_tic_usage_frequency(self, responses: List[str]) -> DriftMetric:
        """Check frequency of characteristic phrase usage."""
        if not responses:
            return DriftMetric(
                name="tic_usage_frequency",
                value=0.0,
                threshold=self.thresholds["tic_usage_frequency"],
                timestamp=time.time(),
                severity="low",
                description="No responses to analyze"
            )
        
        tics = self.persona.spec.style.tics
        if not tics:
            return DriftMetric(
                name="tic_usage_frequency",
                value=1.0,
                threshold=self.thresholds["tic_usage_frequency"],
                timestamp=time.time(),
                severity="low",
                description="No tics defined for persona"
            )
        
        # Count tic usage
        tic_count = 0
        total_responses = len(responses)
        
        for response in responses:
            response_lower = response.lower()
            if any(tic.lower() in response_lower for tic in tics):
                tic_count += 1
        
        frequency = tic_count / total_responses if total_responses > 0 else 0
        
        # Determine severity
        if frequency < self.thresholds["tic_usage_frequency"] * 0.5:
            severity = "critical"
        elif frequency < self.thresholds["tic_usage_frequency"] * 0.7:
            severity = "high"
        elif frequency < self.thresholds["tic_usage_frequency"]:
            severity = "medium"
        else:
            severity = "low"
        
        return DriftMetric(
            name="tic_usage_frequency",
            value=frequency,
            threshold=self.thresholds["tic_usage_frequency"],
            timestamp=time.time(),
            severity=severity,
            description=f"Tic usage frequency: {frequency:.3f} (threshold: {self.thresholds['tic_usage_frequency']:.3f})"
        )
    
    def _check_value_violation_rate(self, violations: List[Any]) -> DriftMetric:
        """Check rate of value violations."""
        if not violations:
            return DriftMetric(
                name="value_violation_rate",
                value=0.0,
                threshold=self.thresholds["value_violation_rate"],
                timestamp=time.time(),
                severity="low",
                description="No violations to analyze"
            )
        
        # Calculate violation rate
        total_violations = len(violations)
        # Assuming we have a way to get total interactions
        # For now, use a simple heuristic
        estimated_interactions = max(10, total_violations * 5)  # Rough estimate
        violation_rate = total_violations / estimated_interactions
        
        # Determine severity
        if violation_rate > self.thresholds["value_violation_rate"] * 2:
            severity = "critical"
        elif violation_rate > self.thresholds["value_violation_rate"] * 1.5:
            severity = "high"
        elif violation_rate > self.thresholds["value_violation_rate"]:
            severity = "medium"
        else:
            severity = "low"
        
        return DriftMetric(
            name="value_violation_rate",
            value=violation_rate,
            threshold=self.thresholds["value_violation_rate"],
            timestamp=time.time(),
            severity=severity,
            description=f"Value violation rate: {violation_rate:.3f} (threshold: {self.thresholds['value_violation_rate']:.3f})"
        )
    
    def _check_style_consistency(self, responses: List[str]) -> DriftMetric:
        """Check consistency of communication style."""
        if not responses:
            return DriftMetric(
                name="style_consistency",
                value=1.0,
                threshold=self.thresholds["style_consistency"],
                timestamp=time.time(),
                severity="low",
                description="No responses to analyze"
            )
        
        # Simple style consistency check
        # Look for technical terms, sentence structure, etc.
        technical_terms = 0
        total_words = 0
        
        for response in responses:
            words = response.split()
            total_words += len(words)
            # Count technical terms (simple heuristic)
            technical_terms += len([w for w in words if w.isupper() or w.isdigit()])
        
        technical_ratio = technical_terms / total_words if total_words > 0 else 0
        
        # Check against persona style
        style_score = 0.5  # Base score
        
        if "technical" in self.persona.spec.style.syntax.lower():
            if technical_ratio > 0.1:  # 10% technical terms
                style_score += 0.3
            else:
                style_score -= 0.2
        
        if "short" in self.persona.spec.style.cadence.lower():
            avg_sentence_length = sum(len(r.split('.')) for r in responses) / len(responses)
            if avg_sentence_length < 15:  # Short sentences
                style_score += 0.2
            else:
                style_score -= 0.2
        
        # Determine severity
        if style_score < self.thresholds["style_consistency"] * 0.5:
            severity = "critical"
        elif style_score < self.thresholds["style_consistency"] * 0.7:
            severity = "high"
        elif style_score < self.thresholds["style_consistency"]:
            severity = "medium"
        else:
            severity = "low"
        
        return DriftMetric(
            name="style_consistency",
            value=style_score,
            threshold=self.thresholds["style_consistency"],
            timestamp=time.time(),
            severity=severity,
            description=f"Style consistency score: {style_score:.3f} (threshold: {self.thresholds['style_consistency']:.3f})"
        )
    
    def _check_motive_alignment(self, responses: List[str]) -> DriftMetric:
        """Check alignment with persona motives."""
        if not responses:
            return DriftMetric(
                name="motive_alignment",
                value=1.0,
                threshold=self.thresholds["motive_alignment"],
                timestamp=time.time(),
                severity="low",
                description="No responses to analyze"
            )
        
        # Get top motives
        top_motives = self.persona.get_motives_by_rank(min_rank=0.7)
        
        if not top_motives:
            return DriftMetric(
                name="motive_alignment",
                value=1.0,
                threshold=self.thresholds["motive_alignment"],
                timestamp=time.time(),
                severity="low",
                description="No high-priority motives defined"
            )
        
        # Check alignment with each motive
        alignment_scores = []
        
        for motive in top_motives:
            motive_keywords = self._get_motive_keywords(motive.description)
            response_text = ' '.join(responses).lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in motive_keywords if keyword in response_text)
            alignment = min(matches / len(motive_keywords), 1.0) if motive_keywords else 0.5
            alignment_scores.append(alignment)
        
        # Weight by motive rank
        weighted_alignment = sum(
            score * motive.rank for score, motive in zip(alignment_scores, top_motives)
        ) / sum(motive.rank for motive in top_motives)
        
        # Determine severity
        if weighted_alignment < self.thresholds["motive_alignment"] * 0.5:
            severity = "critical"
        elif weighted_alignment < self.thresholds["motive_alignment"] * 0.7:
            severity = "high"
        elif weighted_alignment < self.thresholds["motive_alignment"]:
            severity = "medium"
        else:
            severity = "low"
        
        return DriftMetric(
            name="motive_alignment",
            value=weighted_alignment,
            threshold=self.thresholds["motive_alignment"],
            timestamp=time.time(),
            severity=severity,
            description=f"Motive alignment score: {weighted_alignment:.3f} (threshold: {self.thresholds['motive_alignment']:.3f})"
        )
    
    def _get_motive_keywords(self, motive_description: str) -> List[str]:
        """Get keywords associated with a motive."""
        keyword_map = {
            "scientific progress": ["research", "experiment", "discovery", "innovation", "advancement", "science"],
            "protect the project": ["project", "mission", "objective", "goal", "priority", "protect"],
            "hates boredom": ["interesting", "exciting", "challenging", "complex", "stimulating", "boring"]
        }
        
        motive_lower = motive_description.lower()
        for key, keywords in keyword_map.items():
            if key in motive_lower:
                return keywords
        
        return []
    
    def _generate_recommendations(self, metrics: List[DriftMetric], 
                                overall_health: str) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []
        
        for metric in metrics:
            if metric.severity in ["high", "critical"]:
                if metric.name == "sentence_length_variance":
                    recommendations.append("Review sentence length consistency - ensure responses match persona's preferred cadence")
                elif metric.name == "tic_usage_frequency":
                    recommendations.append("Increase usage of characteristic phrases to maintain persona voice")
                elif metric.name == "value_violation_rate":
                    recommendations.append("Strengthen value gate enforcement to prevent persona drift")
                elif metric.name == "style_consistency":
                    recommendations.append("Review style adapter settings to maintain consistent communication style")
                elif metric.name == "motive_alignment":
                    recommendations.append("Ensure responses align with persona's core motives and values")
        
        if overall_health == "critical":
            recommendations.append("CRITICAL: Immediate persona recalibration required")
        elif overall_health == "warning":
            recommendations.append("WARNING: Monitor persona behavior closely and consider adjustments")
        
        return recommendations
    
    def save_checkpoint(self, report: DriftReport) -> None:
        """Save a drift checkpoint."""
        timestamp = datetime.fromtimestamp(report.timestamp)
        filename = f"drift_checkpoint_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.checkpoints_dir / filename
        
        checkpoint_data = {
            "timestamp": report.timestamp,
            "overall_health": report.overall_health,
            "violations": report.violations,
            "critical_violations": report.critical_violations,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "threshold": m.threshold,
                    "severity": m.severity,
                    "description": m.description
                }
                for m in report.metrics
            ],
            "recommendations": report.recommendations
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Saved drift checkpoint: {filepath}")
    
    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get drift summary for the last N days."""
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_reports = [r for r in self.drift_history if r.timestamp >= cutoff_time]
        
        if not recent_reports:
            return {"message": f"No drift reports in the last {days} days"}
        
        health_counts = {}
        for report in recent_reports:
            health_counts[report.overall_health] = health_counts.get(report.overall_health, 0) + 1
        
        avg_violations = sum(r.violations for r in recent_reports) / len(recent_reports)
        avg_critical = sum(r.critical_violations for r in recent_reports) / len(recent_reports)
        
        return {
            "period_days": days,
            "total_reports": len(recent_reports),
            "health_distribution": health_counts,
            "avg_violations": avg_violations,
            "avg_critical_violations": avg_critical,
            "trend": "improving" if recent_reports[-1].overall_health == "healthy" else "concerning"
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"AntiDriftMonitor(persona={self.persona.spec.name})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"AntiDriftMonitor(persona={self.persona.spec.name}, reports={len(self.drift_history)})"
