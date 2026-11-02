"""Workflow regressions; synthetic audio and stub transcription avoid model downloads."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from streamlit.testing.v1 import AppTest
import app


def analysis_page():
    import app
    app.init_session_state()
    app.page_analyze_session()


def full_app():
    import app
    app.main()


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = patch.object(app, 'DB_PATH', str(self.root / 'test.db'))
        self.db.start()
        self.addCleanup(self.db.stop)
        app.init_db()
        pid = app.db_create_patient('Test', 30, 'Other')
        self.audio = self.root / 'speech.wav'
        sf.write(self.audio, 0.1 * np.sin(2 * np.pi * 180 * np.arange(16000) / 16000), 16000)
        self.sid = app.db_create_session(pid, 'Test', str(self.audio), 1, 16000)

    def test_upload_is_reused_and_empty_upload_rejected(self):
        with patch.object(app, 'AUDIO_STORE_DIR', str(self.root)):
            first = app._save_uploaded_audio(self.audio.read_bytes())
            stamp = Path(first).stat().st_mtime_ns
            self.assertEqual(first, app._save_uploaded_audio(self.audio.read_bytes()))
            self.assertEqual(stamp, Path(first).stat().st_mtime_ns)
            with self.assertRaises(ValueError):
                app._save_uploaded_audio(b'')

    def test_pipeline_persists_results(self):
        transcript = dict(raw_transcript='hello world', words=[
            dict(word='hello', start=0.1, end=0.4, confidence=0.9),
            dict(word='world', start=0.5, end=0.9, confidence=0.9)],
            sentences=[], language='en', language_probability=1)
        with patch.object(app, 'transcribe_audio', return_value=transcript), patch.object(app, 'GPT_API_KEY', ''):
            result = app.run_full_analysis_pipeline(self.sid)
        self.assertIn('fluency_score', result)
        self.assertEqual(app.db_get_session(self.sid)['status'], 'completed')
        self.assertIsNotNone(app.db_get_statistics(self.sid))
        self.assertEqual(app.db_get_transcript(self.sid)['raw_transcript'], 'hello world')
        # Render the completed workspace, including every transcript tab.
        at = AppTest.from_function(analysis_page).run(timeout=30)
        self.assertEqual(len(at.exception), 0, str(at.exception))
        words = next(select for select in at.selectbox
                     if select.label == 'Select Transcribed Word to Highlight Segment')
        self.assertEqual(words.options, ["'hello' (0.10s – 0.40s)", "'world' (0.50s – 0.90s)"])


    def test_visible_start_button_and_failed_run(self):
        with patch.object(app, 'FASTER_WHISPER_AVAILABLE', True), patch.object(app, 'run_full_analysis_pipeline', side_effect=RuntimeError('Test failure')):
            at = AppTest.from_function(analysis_page).run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            at.button(key='start_session_analysis').click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            self.assertEqual(app.db_get_session(self.sid)['status'], 'failed')

    def test_upload_starts_analysis_after_navigation(self):
        upload = io.BytesIO(self.audio.read_bytes())
        upload.name = 'sample.wav'
        with patch.object(app, 'AUDIO_STORE_DIR', str(self.root)), \
             patch.object(app.st, 'file_uploader', return_value=upload), \
             patch.object(app, 'FASTER_WHISPER_AVAILABLE', True), \
             patch.object(app, 'run_full_analysis_pipeline', side_effect=RuntimeError('Controlled stop')) as run:
            at = AppTest.from_function(full_app)
            at.session_state['nav_page'] = 'New Session'
            at.run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            button = next(b for b in at.button if b.label == '🔬 Start Analysis')
            button.click().run(timeout=30)
            self.assertEqual(len(at.exception), 0)
            run.assert_called_once()
            self.assertEqual(at.session_state['nav_page'], 'Analyze Session')

    def test_memo_reuses_results_and_refreshes(self):
        stage = 'test_' + self.tmp.name
        calls = []
        def compute():
            calls.append(1)
            return {'words': ['hello'], 'score': 42}
        first = app._session_memo(stage, 'audio-a', compute)
        self.assertEqual(first, app._session_memo(stage, 'audio-a', compute))
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, app._session_memo(stage, 'audio-a', compute, refresh=True))
        self.assertEqual(len(calls), 2)
        app._session_memo(stage, 'audio-b', compute)
        app._session_memo(stage, 'audio-c', compute)
        self.assertEqual(len(app.st.session_state.pipeline_memo[stage]), 2)

    def test_changed_audio_is_decoded_again(self):
        with patch.object(app.librosa, 'load', return_value=(np.ones(20), 16000)) as decode:
            app.load_audio(str(self.audio))
            app.load_audio(str(self.audio))
            self.assertEqual(decode.call_count, 1)
            self.audio.write_bytes(b'changed recording')
            app.load_audio(str(self.audio))
            self.assertEqual(decode.call_count, 2)

    def test_failed_computations_are_not_cached(self):
        from unittest.mock import Mock
        compute = Mock(side_effect=[RuntimeError('temporary failure'), 7])
        with self.assertRaises(RuntimeError):
            app._session_memo('test_failure', self.tmp.name, compute)
        self.assertEqual(app._session_memo('test_failure', self.tmp.name, compute), 7)
        self.assertEqual(compute.call_count, 2)

    def test_missing_engine_does_not_return_fake_transcript(self):
        with patch.object(app, 'load_whisper_model', return_value=None):
            with self.assertRaises(RuntimeError):
                app.transcribe_audio(str(self.audio))


if __name__ == '__main__':
    unittest.main()
