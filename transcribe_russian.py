#!/usr/bin/env python3
"""
Russian Media Transcription and Summarization Script
Supports audio and video files with Russian speech recognition using LOCAL models only
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import json
from datetime import datetime
import torch

# SSL configuration
from ssl_config import configure_ssl_for_self_signed

# Audio/Video processing
import ffmpeg
import whisper

# Text summarization
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Progress bar
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RussianTranscriber:
    """Handles transcription of Russian audio/video files using local models only"""
    
    # Model size configurations
    WHISPER_SIZES = {
        'tiny': {'size': '39M', 'params': '39M', 'english': False, 'multilingual': True},
        'base': {'size': '74M', 'params': '74M', 'english': False, 'multilingual': True},
        'small': {'size': '244M', 'params': '244M', 'english': False, 'multilingual': True},
        'medium': {'size': '769M', 'params': '769M', 'english': False, 'multilingual': True},
        'large': {'size': '1550M', 'params': '1550M', 'english': False, 'multilingual': True}
    }
    
    SUMMARIZER_MODELS = {
        'tiny': 'cointegrated/rut5-small',  # 85M parameters
        'small': 'cointegrated/rut5-small',  # 85M parameters  
        'base': 'IlyaGusev/rut5_base_sum_gazeta',  # 223M parameters
        'large': 'IlyaGusev/rut5_base_sum_gazeta'  # 223M parameters
    }
    
    def __init__(self, model_size: str = "small", model_dir: Optional[Path] = None, 
                 device: str = "auto", lightweight: bool = False, enable_summarization: bool = False):
        """
        Initialize the transcriber with specified Whisper model
        
        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            model_dir: Directory to store/load models locally
            device: Device to use ('cpu', 'cuda', 'auto')
            lightweight: Use smaller summarization model to save memory
            enable_summarization: Whether to load summarization models (default: False)
        """
        # Setup model directory
        if model_dir is None:
            model_dir = Path.home() / ".cache" / "russian_transcriber"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model locally
        whisper_info = self.WHISPER_SIZES.get(model_size, self.WHISPER_SIZES['small'])
        logger.info(f"Loading Whisper model: {model_size} ({whisper_info['size']}) locally")
        whisper_cache = self.model_dir / "whisper"
        whisper_cache.mkdir(exist_ok=True)
        os.environ['WHISPER_CACHE'] = str(whisper_cache)
        self.model = whisper.load_model(model_size, download_root=str(whisper_cache))
        
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.mp4', '.avi', '.mkv', '.mov', '.webm'}
        
        # Initialize summarization components
        self.summarizer = None
        self.use_fallback_summarizer = False
        self.enable_summarization = enable_summarization
        
        if enable_summarization:
            logger.info("📝 Initializing summarization...")
            self._load_summarization_models(lightweight, model_size)
        else:
            logger.info("⏭️  Summarization disabled (transcript only mode) - use --enable-summary to enable")
    
    def _load_summarization_models(self, lightweight: bool, model_size: str):
        """Load summarization models with fallback"""
        try:
            # Choose summarizer based on model size or lightweight flag
            if lightweight or model_size in ['tiny', 'small']:
                self.summarizer_model_name = self.SUMMARIZER_MODELS['tiny']
                logger.info("Loading lightweight Russian summarization model (rut5-small)")
            else:
                self.summarizer_model_name = self.SUMMARIZER_MODELS['base']
                logger.info("Loading standard Russian summarization model (rut5-base)")
                
            summarizer_cache = self.model_dir / "summarizer"
            summarizer_cache.mkdir(exist_ok=True)
            
            # Load with local cache
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.summarizer_model_name,
                cache_dir=summarizer_cache,
                local_files_only=False  # Will download first time, then use local
            )
            self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.summarizer_model_name,
                cache_dir=summarizer_cache,
                local_files_only=False,  # Will download first time, then use local
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move model to device
            if self.device == "cuda":
                self.summarizer_model = self.summarizer_model.to(self.device)
            
            self.summarizer = pipeline(
                "summarization",
                model=self.summarizer_model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to load transformer summarization model: {e}")
            logger.info("🔄 Falling back to simple rule-based summarizer...")
            self.use_fallback_summarizer = True
            self._setup_fallback_summarizer()
    
    def _setup_fallback_summarizer(self):
        """Setup simple rule-based Russian summarizer"""
        import re
        
        # Russian stop words
        self.stop_words = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так',
            'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было',
            'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг',
            'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж',
            'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
            'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего',
            'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого',
            'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас',
            'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть',
            'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
            'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть',
            'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }
        logger.info("✓ Fallback summarizer initialized")
    
    def extract_audio(self, input_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Extract audio from video file or copy audio file
        
        Args:
            input_path: Path to input media file
            output_path: Optional path for extracted audio
            
        Returns:
            Path to audio file
        """
        if output_path is None:
            output_path = input_path.with_suffix('.wav')
        
        try:
            logger.info(f"Extracting audio from: {input_path}")
            stream = ffmpeg.input(str(input_path))
            stream = ffmpeg.output(stream, str(output_path), 
                                 acodec='pcm_s16le', 
                                 ac=1, 
                                 ar='16k')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            logger.info(f"Audio extracted to: {output_path}")
            return output_path
        except ffmpeg.Error as e:
            logger.error(f"Error extracting audio: {e}")
            raise
    
    def transcribe(self, audio_path: Path, language: str = "ru") -> dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_path: Path to audio file
            language: Language code (default: 'ru' for Russian)
            
        Returns:
            Transcription result dictionary
        """
        logger.info(f"🎙️  Starting transcription of: {audio_path}")
        
        # Get audio duration for progress estimation
        try:
            import librosa
            duration = librosa.get_duration(filename=str(audio_path))
            logger.info(f"📏 Audio duration: {duration:.1f} seconds")
        except:
            duration = None
        
        # Enhanced progress tracking for transcription
        from tqdm import tqdm
        import time
        import threading
        
        # Progress tracking variables  
        progress_bar = None
        transcription_done = threading.Event()
        
        def show_enhanced_progress():
            """Show enhanced progress bar with better estimation"""
            if duration:
                # Create progress bar with custom format
                progress_bar = tqdm(
                    total=int(duration), 
                    desc="🎙️  Transcribing audio", 
                    unit="sec",
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}] {rate_fmt}',
                    ncols=80,
                    colour='green'
                )
                
                start_time = time.time()
                last_update = 0
                
                while not transcription_done.wait(0.5):  # Update every 0.5 seconds
                    elapsed = time.time() - start_time
                    
                    # Whisper processes roughly in real-time, sometimes faster
                    # Estimate progress based on elapsed time vs audio duration
                    estimated_progress = min(elapsed, duration)
                    
                    if estimated_progress > last_update:
                        progress_bar.n = estimated_progress
                        progress_bar.refresh()
                        last_update = estimated_progress
                
                # Complete the progress bar
                progress_bar.n = progress_bar.total
                progress_bar.refresh()
                progress_bar.close()
            else:
                # Fallback spinner for unknown duration
                spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                i = 0
                while not transcription_done.wait(0.1):
                    print(f"\r🎙️  Transcribing... {spinner_chars[i % len(spinner_chars)]}", end="", flush=True)
                    i += 1
                print(f"\r🎙️  Transcribing... ✅ Done!     ")
        
        # Start enhanced progress thread
        progress_thread = threading.Thread(target=show_enhanced_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Transcribe with Whisper
            result = self.model.transcribe(
                str(audio_path),
                language=language,
                task="transcribe",
                verbose=False,
                fp16=False  # Disable FP16 for CPU
            )
            
        finally:
            # Stop progress tracking
            transcription_done.set()
        
        logger.info(f"✅ Transcription completed. Duration: {result.get('duration', 0):.2f} seconds")
        return result
    
    def summarize_text(self, text: str, max_length: int = 300, min_length: int = 50) -> str:
        """
        Generate summary of Russian text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summary text or empty string if summarization disabled
        """
        if not self.enable_summarization:
            logger.info("⏭️  Summarization disabled")
            return ""
        
        logger.info("Generating summary")
        
        if self.use_fallback_summarizer:
            return self._fallback_summarize(text, max_sentences=3)
        
        try:
            # Split text into chunks if too long
            max_chunk_length = 1000
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            
            summaries = []
            for chunk in tqdm(chunks, desc="Summarizing chunks"):
                summary = self.summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                summaries.append(summary[0]['summary_text'])
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined_text = " ".join(summaries)
                final_summary = self.summarizer(
                    combined_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return final_summary[0]['summary_text']
            
            return summaries[0] if summaries else ""
            
        except Exception as e:
            logger.warning(f"⚠️  Transformer summarization failed: {e}")
            logger.info("🔄 Using fallback summarizer...")
            return self._fallback_summarize(text, max_sentences=3)
    
    def _fallback_summarize(self, text: str, max_sentences: int = 3) -> str:
        """Simple extractive summarization for Russian text"""
        import re
        
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Get word frequencies
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize frequencies
        if word_freq:
            max_freq = max(word_freq.values())
            for word in word_freq:
                word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words_in_sent = re.findall(r'\b\w+\b', sentence.lower())
            words_in_sent = [w for w in words_in_sent if w not in self.stop_words and len(w) > 2]
            
            if not words_in_sent:
                score = 0
            else:
                score = sum(word_freq.get(word, 0) for word in words_in_sent) / len(words_in_sent)
            
            sentence_scores.append((score, i, sentence))
        
        # Sort by score and take top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = sentence_scores[:max_sentences]
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x[1])
        
        # Join selected sentences
        summary = '. '.join([sent[2] for sent in top_sentences])
        return summary + '.' if summary else text[:500] + "..."
    
    def process_media_file(self, input_path: Path, output_dir: Optional[Path] = None) -> Tuple[str, str]:
        """
        Process media file: extract audio, transcribe, and summarize
        
        Args:
            input_path: Path to input media file
            output_dir: Optional output directory for results
            
        Returns:
            Tuple of (transcript, summary)
        """
        # Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if input_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported format: {input_path.suffix}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract audio if video
        temp_audio = None
        if input_path.suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov', '.webm'}:
            temp_audio = output_dir / f"{input_path.stem}_temp.wav"
            audio_path = self.extract_audio(input_path, temp_audio)
        else:
            audio_path = input_path
        
        try:
            # Transcribe
            transcription_result = self.transcribe(audio_path)
            transcript = transcription_result['text']
            
            # Generate summary
            summary = self.summarize_text(transcript)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{input_path.stem}_{timestamp}"
            
            # Save transcript
            transcript_path = output_dir / f"{base_name}_transcript.txt"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            logger.info(f"Transcript saved to: {transcript_path}")
            
            # Save summary if enabled
            if self.enable_summarization and summary:
                summary_path = output_dir / f"{base_name}_summary.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info(f"Summary saved to: {summary_path}")
            
            # Save full results as JSON
            results_path = output_dir / f"{base_name}_results.json"
            results = {
                'input_file': str(input_path),
                'timestamp': timestamp,
                'duration': transcription_result.get('duration', 0),
                'language': transcription_result.get('language', 'ru'),
                'transcript': transcript,
                'summarization_enabled': self.enable_summarization,
                'segments': transcription_result.get('segments', [])
            }
            
            # Only add summary if enabled and generated
            if self.enable_summarization:
                results['summary'] = summary
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Full results saved to: {results_path}")
            
            return transcript, summary
            
        finally:
            # Cleanup temporary audio file
            if temp_audio and temp_audio.exists():
                temp_audio.unlink()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Transcribe and summarize Russian audio/video files using LOCAL models"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input audio/video file path"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "-m", "--model",
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='small',
        help="Whisper model size (default: small)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Directory to store/load models (default: ~/.cache/russian_transcriber)"
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help="Device to use for inference (default: auto)"
    )
    parser.add_argument(
        "--max-summary-length",
        type=int,
        default=300,
        help="Maximum summary length in tokens (default: 300)"
    )
    parser.add_argument(
        "--min-summary-length",
        type=int,
        default=50,
        help="Minimum summary length in tokens (default: 50)"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only locally cached models (no downloads)"
    )
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Use lightweight models to reduce memory usage"
    )
    parser.add_argument(
        "--enable-summary",
        action="store_true",
        help="Enable summarization (requires additional model downloads)"
    )
    
    args = parser.parse_args()
    
    try:
        # Configure SSL if needed
        configure_ssl_for_self_signed()
        
        # Set offline mode if requested
        if args.offline:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            logger.info("Running in offline mode - using only cached models")
        
        # Initialize transcriber
        transcriber = RussianTranscriber(
            model_size=args.model,
            model_dir=args.model_dir,
            device=args.device,
            lightweight=args.lightweight,
            enable_summarization=args.enable_summary
        )
        
        # Process file
        transcript, summary = transcriber.process_media_file(
            args.input,
            args.output
        )
        
        # Print results
        print("\n" + "="*50)
        print("TRANSCRIPT:")
        print("="*50)
        print(transcript[:500] + "..." if len(transcript) > 500 else transcript)
        
        if args.enable_summary and summary:
            print("\n" + "="*50)
            print("SUMMARY:")
            print("="*50)
            print(summary)
        elif not args.enable_summary:
            print("\n" + "⏭️  Summarization disabled (use --enable-summary to enable)")
        
        print("="*50 + "\n")
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()