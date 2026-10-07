# ClinicalSpeech AI 🎙️

**Comprehensive Speech Fluency & Stuttering Event Analysis Platform**

ClinicalSpeech AI is an advanced clinical analytics platform designed for Speech-Language Pathologists (SLPs) and researchers. It provides multi-category disfluency detection, acoustic signal analysis, synchronized verbatim and cleaned transcription, interactive speech timelines, and automated clinical report (CVRET) generation.

---

## 🌟 Key Features

- **Clinical Dashboard**: Real-time KPI cards, severity distribution charts, and recent session monitoring.
- **Patient Management**: Full patient registry, clinical history, and session tracking.
- **Guided Intake Pipeline**: 6-step clinical workflow supporting both audio uploads and live microphone recordings.
- **Analytical Workspace**:
  - Speech Overview (Duration, speaking time, silence time, acoustic properties)
  - Fluency Metrics (%SS, WPM, SPM, pause ratios)
  - 16-Category Stuttering Event Detection
  - Speech & Silence Segmentation
  - Interactive Multi-Track Timeline (waveform envelope + event scatter tracks)
  - Synchronized Transcription (Verbatim vs Clean transcript)
  - Clinical Observations (SLP notes & AI-generated summaries)
- **Clinical Reports (CVRET)**: Standardized clinical documentation with waveform overlays and PDF generation.
- **Longitudinal History & Comparison**: Side-by-side session comparisons (Session A vs Session B) with empirical delta status indicators (`Improved`, `Stable`, `Increased Disfluency`).

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.9+ installed and `ffmpeg` available on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
python3 -m streamlit run app.py
```

The application will start at `http://localhost:8501` (or next available port).

---

## 🔬 AI Engines & Dependencies
- **Speech Transcription**: [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- **Acoustic Analysis**: [librosa](https://librosa.org/), [scipy](https://scipy.org/)
- **Phoneme Alignment (Optional)**: [whisperx](https://github.com/m-bain/whisperX)
- **Report Generation**: [reportlab](https://www.reportlab.com/)
- **Transcript Disfluency Cleaning**: OpenAI GPT API (optional fallback to rule-based)

---

## 📄 License
Research and clinical evaluation use.