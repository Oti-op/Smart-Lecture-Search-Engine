import whisper


class Transcriber:
    def __init__(self, model_name: str = "base") -> None:
        """Load the Whisper model. model_name is the Whisper size (tiny/base/small/…)."""
        print(f"[Transcriber] Loading Whisper model '{model_name}'…")
        self._model = whisper.load_model(model_name)
        print(f"[Transcriber] Model '{model_name}' ready.")

    def transcribe(self, audio_path: str) -> dict:
        """Transcribe an audio file and return timestamped segments.

        Args:
            audio_path: Absolute or relative path to the audio file.

        Returns:
            {
                "segments": [{"start": float, "end": float, "text": str}, ...],
                "full_text": str,
            }

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If Whisper fails to process the file.
        """
        import os

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            result = self._model.transcribe(audio_path, fp16=False)
        except Exception as exc:
            raise RuntimeError(
                f"Whisper failed to transcribe '{audio_path}': {exc}"
            ) from exc

        segments = [
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"].strip(),
            }
            for seg in result.get("segments", [])
        ]

        return {
            "segments": segments,
            "full_text": result.get("text", "").strip(),
        }
