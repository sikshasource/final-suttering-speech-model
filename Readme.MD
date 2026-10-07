# ClinicalSpeech AI

### Comprehensive Speech Fluency & Stuttering Event Analysis Platform

ClinicalSpeech AI is a clinical speech-analysis platform designed for
Speech-Language Pathologists (SLPs), speech researchers, and clinical
evaluation workflows.

The platform processes recorded or uploaded speech, extracts acoustic and
fluency characteristics, detects categorized stuttering/disfluency events,
provides synchronized speech timelines and transcripts, and generates
structured clinical reports.

> ClinicalSpeech AI is intended for research and clinical evaluation use.
> It is not a replacement for professional clinical judgment or diagnosis.

---

## Overview

ClinicalSpeech AI combines speech processing, acoustic analysis,
transcription, event detection, segmentation, visualization, and clinical
report generation into a unified workflow.

The system is organized around a patient/session model:

Patient
  ↓
Clinical Session
  ↓
Audio Input
  ↓
Speech Processing
  ├── Transcription
  ├── Acoustic Analysis
  ├── Speech/Silence Segmentation
  ├── Stuttering Event Detection
  └── Optional Phoneme Alignment
  ↓
Analysis Workspace
  ├── Speech Overview
  ├── Fluency Metrics
  ├── Event Timeline
  ├── Segmentation
  ├── Verbatim Transcript
  ├── Clean Transcript
  └── Clinical Observations
  ↓
Clinical Report
  ↓
Longitudinal Comparison

---

# Core Capabilities

## 1. Clinical Dashboard

Provides a high-level operational view of the system.

Includes:

- Total patients
- Total sessions
- Completed analyses
- Pending analyses
- Recent sessions
- Severity distribution
- Session activity over time
- Processing engine status

The dashboard is designed to answer three questions quickly:

1. What has been analyzed?
2. What requires attention?
3. What is the current clinical workload?

---

## 2. Patient Management

The patient module provides the central registry for clinical sessions.

Capabilities include:

- Patient registration
- Patient profile management
- Clinical history
- Session history
- Session-level analysis access
- Longitudinal comparison

The patient entity acts as the parent context for multiple speech-analysis
sessions.

Conceptually:

Patient
  └── Session
       ├── Audio
       ├── Transcription
       ├── Segments
       ├── Stuttering Events
       ├── Metrics
       ├── Observations
       └── Clinical Report

---

## 3. Guided Clinical Intake Pipeline

The application provides a six-step workflow for creating and processing a
clinical speech session.

### Step 1 — Patient

Select an existing patient or create a new patient.

### Step 2 — Session Information

Capture the session metadata supported by the application.

### Step 3 — Audio Input

Supports:

- Audio file upload
- Microphone recording

### Step 4 — Audio Review

Review the captured audio and session information before analysis.

### Step 5 — Analysis

The processing pipeline performs the available speech-analysis operations.

### Step 6 — Results

The user is taken to the analytical workspace where processed results can
be reviewed.

The workflow is intentionally staged so that raw audio, processing,
analysis, and clinical interpretation remain distinguishable.

---

# Analytical Workspace

The analytical workspace is the core of ClinicalSpeech AI.

It combines quantitative speech metrics with event-level and
time-synchronized analysis.

## Speech Overview

Provides the available high-level speech characteristics, including:

- Total duration
- Speaking time
- Silence time
- Acoustic properties

The exact metrics displayed depend on the analysis output available for the
session.

---

## Fluency Metrics

The platform supports fluency-oriented metrics including:

- Percentage of syllables stuttered (%SS)
- Words per minute (WPM)
- Syllables per minute (SPM)
- Pause ratios

Metrics should always be interpreted in their clinical context rather than
as standalone diagnostic conclusions.

---

## Stuttering Event Detection

ClinicalSpeech AI supports detection and categorization across 16
stuttering/disfluency event categories.

The event model is designed to preserve event-level information rather than
reducing an entire recording to a single severity value.

Each detected event can be associated with temporal information and the
corresponding analytical context.

---

## Speech & Silence Segmentation

The system separates speech and silence regions to support:

- Speech-duration analysis
- Pause analysis
- Timeline visualization
- Fluency calculations
- Clinical interpretation

Segmentation provides the temporal foundation for downstream analysis.

---

## Interactive Speech Timeline

The timeline provides synchronized visualization of speech activity and
detected events.

The analytical timeline can combine:

- Audio waveform/envelope
- Speech regions
- Silence regions
- Stuttering-event markers
- Event categories
- Temporal positions

The goal is to allow the clinician or researcher to move from aggregate
metrics to the underlying speech events that produced those metrics.

---

## Synchronized Transcription

ClinicalSpeech AI maintains two transcript representations where available:

### Verbatim Transcript

Represents the detected speech as closely as possible to the transcription
output.

### Clean Transcript

Provides a cleaned representation intended for easier clinical reading.

The two representations should remain distinguishable.

Cleaning is performed through the configured GPT-based processing path when
available, with a rule-based fallback.

---

## Clinical Observations

The workspace supports:

- Clinician/SLP observations
- AI-generated summaries where configured

AI-generated content should be treated as assistive output and should not
automatically be represented as clinician-authored findings.

---

# Clinical Reporting

ClinicalSpeech AI provides standardized clinical report generation using
the application's CVRET reporting workflow.

Reports can incorporate:

- Patient information
- Session information
- Speech metrics
- Fluency results
- Stuttering-event information
- Timeline information
- Waveform overlays
- Clinical observations
- Generated analysis content

The reporting pipeline uses ReportLab for PDF generation.

---

# Longitudinal History & Comparison

ClinicalSpeech AI supports comparison between speech-analysis sessions.

A typical comparison follows:

Session A
    ↓
Baseline measurements
    ↓
Session B
    ↓
Current measurements
    ↓
Metric delta
    ↓
Comparison status

Supported comparison states include:

- Improved
- Stable
- Increased Disfluency

These indicators should represent calculated differences between available
measurements and should not be interpreted as autonomous clinical diagnoses.

---

# AI & Signal Processing Stack

The application combines several specialized processing components.

| Component | Technology | Responsibility |
|---|---|---|
| Application UI | Streamlit | Clinical application interface |
| Speech transcription | faster-whisper | Speech-to-text processing |
| Acoustic analysis | librosa | Audio/signal feature extraction |
| Scientific processing | scipy | Signal processing and numerical analysis |
| Phoneme alignment | WhisperX | Optional alignment workflow |
| PDF generation | ReportLab | Clinical report generation |
| Transcript cleaning | OpenAI GPT API | Optional transcript cleaning/summarization |
| Transcript fallback | Rule-based processing | Deterministic fallback when GPT is unavailable |

Optional components should degrade gracefully where possible.

For example:

GPT cleaning unavailable
        ↓
Rule-based cleaning
        ↓
Continue workflow

The fallback must never silently claim that an AI-generated transformation
was performed when it was not.

---

# System Architecture

The current application is a Python/Streamlit system.

At a high level:

```text
┌───────────────────────────────────────────────┐
│                Streamlit UI                   │
│                                               │
│ Dashboard                                     │
│ Patients                                      │
│ New Session                                   │
│ Analyze Session                               │
│ Timeline                                      │
│ Segmentation                                  │
│ Statistics                                    │
│ Clinical Report                               │
│ Patient History & Compare                     │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│              Application Layer                │
│                                               │
│ Session Management                            │
│ Patient Management                            │
│ Analysis Orchestration                        │
│ Report Orchestration                          │
│ Comparison Logic                              │
└──────────────────────┬────────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────────┐
│              Processing Layer                 │
│                                               │
│ Audio Processing                              │
│ ├── Audio Validation                          │
│ ├── Pre-processing                            │
│ └── Feature Extraction                        │
│                                               │
│ Speech Processing                             │
│ ├── Transcription                             │
│ ├── Alignment                                 │
│ └── Segmentation                              │
│                                               │
│ Fluency Analysis                              │
│ ├── Speech Metrics                            │
│ ├── Pause Metrics                             │
│ └── Disfluency Metrics                        │
│                                               │
│ Event Analysis                                │
│ └── 16-category Event Detection               │
└──────────────────────┬────────────────────────┘
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
┌──────────────────────┐  ┌─────────────────────┐
│ Analysis Results     │  │ Clinical Reporting  │
│                      │  │                     │
│ Metrics              │  │ CVRET               │
│ Events               │  │ Waveform overlays   │
│ Segments             │  │ PDF generation      │
│ Transcripts          │  │                     │
│ Observations         │  │ ReportLab           │
└──────────────────────┘  └─────────────────────┘
