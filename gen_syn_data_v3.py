#!/usr/bin/env python3
"""
Synthetic TTS Data Generator v3 - Client/Server Architecture
Uses vLLM server (running separately) + OpenAI client for inference
Codec runs on separate GPU controlled by CUDA_VISIBLE_DEVICES
"""

import os
import sys
import random
import json
import argparse
import psutil
import gc
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from datasets import load_dataset, Dataset
from loguru import logger
import re
import time

from openai import OpenAI
from phonemizer import phonemize
from neucodec import NeuCodec


@dataclass
class SyntheticSample:
    """Metadata for a synthetic sample"""
    sample_id: str
    reference_audio: str
    reference_text: str
    target_text: str
    target_text_idx: int
    output_audio: str
    duration_sec: float
    reference_speaker_id: Optional[str] = None


class VLLMClientSyntheticGenerator:
    """Generate synthetic TTS data using vLLM server via OpenAI client."""
    
    def __init__(
        self,
        vllm_server_url: str,
        model_name: str,
        output_dir: str,
        codec_repo: str = "neuphonic/neucodec",
        sample_rate: int = 24000,
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 4096
    ):
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.model_name = model_name
        
        # Codec device controlled by CUDA_VISIBLE_DEVICES
        self.codec_device = "cuda:0"
        
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        
        # Create output structure
        self.audio_dir = self.output_dir / "audio"
        self.metadata_dir = self.output_dir / "metadata"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("VLLM CLIENT GENERATOR v3")
        logger.info(f"Server: {vllm_server_url}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Codec Device: {self.codec_device} (via CUDA_VISIBLE_DEVICES)")
        
        # Log system memory
        mem = psutil.Process().memory_info()
        logger.info(f"Initial Memory: RSS={mem.rss/1024**3:.2f}GB, VMS={mem.vms/1024**3:.2f}GB")
        
        # Log GPU memory
        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU Memory: {gpu_mem_alloc:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved, {gpu_mem_total:.2f}GB total")
        logger.info("=" * 80)
        
        # Initialize OpenAI client
        logger.info("Connecting to vLLM server...")
        self.client = OpenAI(base_url=vllm_server_url, api_key="EMPTY")
        
        try:
            models = self.client.models.list()
            logger.success(f"✓ Connected! Available models: {[m.id for m in models.data]}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
        
        # Load codec (same way as neutts.py)
        logger.info(f"Loading codec from {codec_repo} on {self.codec_device}...")
        mem_before = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"System RAM before codec load: {mem_before:.2f}GB")
        
        if torch.cuda.is_available():
            gpu_before = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"GPU memory before codec load: {gpu_before:.2f}GB")
        
        self.codec = NeuCodec.from_pretrained(codec_repo)
        self.codec.eval().to(self.codec_device)
        
        mem_after = psutil.Process().memory_info().rss / 1024**3
        logger.success(f"✓ Codec loaded on {self.codec_device}")
        logger.info(f"System RAM after codec load: {mem_after:.2f}GB (delta: +{mem_after-mem_before:.2f}GB)")
        
        if torch.cuda.is_available():
            gpu_after = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"GPU memory after codec load: {gpu_after:.2f}GB allocated, {gpu_reserved:.2f}GB reserved (delta: +{gpu_after-gpu_before:.2f}GB)")
        
        self.reference_cache: Dict[str, List[int]] = {}
        
        mem_final = psutil.Process().memory_info().rss / 1024**3
        logger.success("Initialization complete!")
        logger.info(f"Total system RAM usage: {mem_final:.2f}GB")
        
        if torch.cuda.is_available():
            gpu_final = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved_final = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"Total GPU memory: {gpu_final:.2f}GB allocated, {gpu_reserved_final:.2f}GB reserved")
    
    def to_phones(self, text: str) -> str:
        """Convert text to phonemes using espeak."""
        return phonemize(text, language='ar', backend='espeak', strip=True, 
                        preserve_punctuation=True, with_stress=False)
    
    def encode_reference_audio(self, audio_data: np.ndarray, sample_rate: int) -> List[int]:
        """Encode reference audio to codes (with caching)."""
        cache_key = str(hash(audio_data.tobytes()))
        if cache_key in self.reference_cache:
            logger.debug(f"Cache hit for audio (cache size: {len(self.reference_cache)})")
            return self.reference_cache[cache_key]
        
        logger.debug(f"Encoding reference audio: {len(audio_data)} samples @ {sample_rate}Hz")
        
        # Log GPU memory before encoding
        if torch.cuda.is_available():
            gpu_before_encode = torch.cuda.memory_allocated(0) / 1024**3
            logger.debug(f"GPU memory before encode: {gpu_before_encode:.2f}GB")
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            audio_tensor = torch.from_numpy(audio_data).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            resampled = resampler(audio_tensor)
            # Ensure tensor is on CPU before converting to numpy
            audio_data = resampled.squeeze(0).cpu().numpy()
        
        # Encode
        # Following neutts.py pattern (line 191-193):
        # Create tensor [1, 1, T] on CPU, codec handles device placement internally
        logger.debug(f"Encoding audio: shape={audio_data.shape}, dtype={audio_data.dtype}")
        
        wav_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T] on CPU
        logger.debug(f"Audio tensor for codec: shape={wav_tensor.shape}, device={wav_tensor.device}")
        
        with torch.no_grad():
            ref_codes_tensor = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        
        # Free tensors
        del wav_tensor
        
        # Move codes to CPU before converting to list
        ref_codes = ref_codes_tensor.cpu().tolist() if hasattr(ref_codes_tensor, 'tolist') else [int(x) for x in ref_codes_tensor.cpu()]
        
        # Free tensor
        del ref_codes_tensor
        if torch.cuda.is_available():
            gpu_after_encode = torch.cuda.memory_allocated(0) / 1024**3
            logger.debug(f"GPU memory after encode: {gpu_after_encode:.2f}GB")
        
        self.reference_cache[cache_key] = ref_codes
        logger.debug(f"Encoded to {len(ref_codes)} codes, cache size: {len(self.reference_cache)}")
        return ref_codes
    
    def build_prompt(self, target_phonemes: str, ref_phonemes: str, ref_codes: List[int]) -> str:
        """Build inference prompt matching training format.
        
        Following neutts.py line 282-285:
        - TEXT_PROMPT contains: ref_text + target_text (combined)
        - SPEECH_GENERATION starts with ref_codes only
        - Model continues generating codes for the full combined text
        """
        # Combine ref and target phonemes (ref first, then target)
        combined_phonemes = f"{ref_phonemes} {target_phonemes}"
        
        # Reference codes as string (model will continue from here)
        ref_codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])
        
        # Build prompt exactly like neutts.py
        prompt = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{combined_phonemes}"
            f"<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{ref_codes_str}"
        )
        
        logger.debug(f"Prompt components:")
        logger.debug(f"  ref_phonemes: {len(ref_phonemes)} chars")
        logger.debug(f"  target_phonemes: {len(target_phonemes)} chars")
        logger.debug(f"  combined_phonemes: {len(combined_phonemes)} chars")
        logger.debug(f"  ref_codes: {len(ref_codes)} codes")
        logger.debug(f"  ref_codes_str: {len(ref_codes_str)} chars")
        logger.debug(f"  Total prompt: {len(prompt)} chars")
        logger.debug(f"  Prompt ends with: ...{prompt[-100:]}")
        
        return prompt
    
    def generate_codes(self, prompt: str) -> List[int]:
        """Generate speech codes via vLLM server using OpenAI Completions API."""
        logger.debug(f"Calling vLLM (prompt: {len(prompt)} chars)")
        logger.debug(f"Prompt preview (last 500 chars): ...{prompt[-500:]}")
        
        start = time.time()
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["<|SPEECH_GENERATION_END|>"],
            echo=False,  # Keep False at API level
            extra_body={
                "skip_special_tokens": False,  # CRITICAL: Keep speech tokens like <|speech_123|>
                "echo": True  # Return prompt + generation to debug
            }
        )
        logger.debug(f"Generated in {time.time()-start:.2f}s")
        
        # Debug: Log full response object
        logger.debug(f"Response object: {response}")
        logger.debug(f"Response choices: {len(response.choices)}")
        logger.debug(f"Response finish_reason: {response.choices[0].finish_reason}")
        
        # Extract speech codes
        generated = response.choices[0].text
        logger.debug(f"Generated text length: {len(generated)}")
        logger.debug(f"Generated text type: {type(generated)}")
        logger.debug(f"Generated text full: '{generated}'")
        logger.debug(f"Generated text repr: {repr(generated)}")
        
        matches = re.findall(r'<\|speech_(\d+)\|>', generated)
        codes = [int(m) for m in matches]
        
        logger.debug(f"Regex matches: {matches[:10] if matches else 'None'}")
        logger.debug(f"Extracted {len(codes)} codes")
        
        if len(codes) == 0:
            logger.error(f"No codes extracted!")
            logger.error(f"Prompt used (last 200 chars): ...{prompt[-200:]}")
            logger.error(f"Full response text ({len(generated)} chars): '{generated}'")
            logger.error(f"Stop reason: {response.choices[0].finish_reason}")
            
        return codes
    
    def codes_to_audio(self, codes: List[int]) -> np.ndarray:
        """Decode speech codes to audio waveform."""
        if torch.cuda.is_available():
            gpu_before_decode = torch.cuda.memory_allocated(0) / 1024**3
            logger.debug(f"GPU memory before decode: {gpu_before_decode:.2f}GB")
        
        # Codec expects shape [batch, 1, sequence_length]
        codes_tensor = torch.tensor(codes).unsqueeze(0).unsqueeze(0).to(self.codec_device)
        logger.debug(f"Codes tensor shape: {codes_tensor.shape}, device: {codes_tensor.device}")
        
        with torch.no_grad():
            audio_tensor = self.codec.decode_code(codes_tensor)
        
        audio_np = audio_tensor.squeeze().cpu().numpy()
        
        # Free GPU memory
        del codes_tensor, audio_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_after_decode = torch.cuda.memory_allocated(0) / 1024**3
            logger.debug(f"GPU memory after decode: {gpu_after_decode:.2f}GB")
        
        return audio_np
    
    def generate_sample(
        self,
        ref_audio_data: np.ndarray,
        ref_sample_rate: int,
        ref_text: str,
        target_text: str,
        target_text_idx: int,
        output_path: str,
        speaker_id: Optional[str] = None
    ) -> SyntheticSample:
        """Generate a single synthetic sample."""
        start = time.time()
        
        ref_codes = self.encode_reference_audio(ref_audio_data, ref_sample_rate)
        ref_phonemes = self.to_phones(ref_text)
        target_phonemes = self.to_phones(target_text)
        
        logger.info(f"\nGenerating: '{target_text[:60]}...' (ref codes: {len(ref_codes)})")
        
        prompt = self.build_prompt(target_phonemes, ref_phonemes, ref_codes)
        generated_codes = self.generate_codes(prompt)
        
        if len(generated_codes) == 0:
            raise ValueError("Model generated 0 speech codes")
        
        generated_wav = self.codes_to_audio(generated_codes)
        duration = len(generated_wav) / self.sample_rate
        
        elapsed = time.time() - start
        logger.info(f"✓ {duration:.2f}s audio in {elapsed:.2f}s (RTF: {elapsed/duration:.2f}x)")
        
        sf.write(output_path, generated_wav, self.sample_rate)
        
        return SyntheticSample(
            sample_id=Path(output_path).stem,
            reference_audio="<from_dataset>",
            reference_text=ref_text,
            target_text=target_text,
            target_text_idx=target_text_idx,
            output_audio=output_path,
            duration_sec=round(duration, 3),
            reference_speaker_id=speaker_id
        )
    
    def generate_from_dataset(
        self,
        dataset_path: str,
        target_texts_file: str,
        samples_per_reference: int = 10,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> List[SyntheticSample]:
        """Generate synthetic data from dataset."""
        # Count target texts without loading into memory
        logger.info(f"Counting target texts in: {target_texts_file}")
        mem_before_count = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Memory before counting: {mem_before_count:.2f}GB")
        
        num_target_texts = sum(1 for line in open(target_texts_file, 'r', encoding='utf-8') if line.strip())
        
        mem_after_count = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Found {num_target_texts:,} target texts (lazy loading)")
        logger.info(f"Memory after counting: {mem_after_count:.2f}GB (delta: +{mem_after_count-mem_before_count:.2f}GB)")
        
        logger.info(f"Loading dataset: {dataset_path}")
        mem_before_ds = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Memory before dataset load: {mem_before_ds:.2f}GB")
        
        # Load dataset WITHOUT caching audio in memory
        ds = load_dataset(dataset_path, split="train")
        
        # Force dataset to NOT cache decoded audio
        ds = ds.with_format("python")
        
        mem_after_ds = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Memory after dataset load: {mem_after_ds:.2f}GB (delta: +{mem_after_ds-mem_before_ds:.2f}GB)")
        logger.info(f"Dataset columns: {ds.column_names}")
        logger.info(f"Dataset features: {ds.features}")
        
        if not isinstance(ds, Dataset):
            raise ValueError(f"Expected Dataset, got {type(ds)}")
        
        logger.info(f"Dataset: {len(ds)} samples")
        
        if end_idx is not None:
            ds = ds.select(range(start_idx, min(end_idx, len(ds))))
        else:
            ds = ds.select(range(start_idx, len(ds)))
        
        logger.info(f"Selected range: {start_idx} to {end_idx if end_idx else len(ds)}")
        logger.info(f"Processing {len(ds)} refs → {len(ds) * samples_per_reference} total samples")
        
        # Check memory before selecting
        mem_after_select = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"System RAM after select: {mem_after_select:.2f}GB")
        mem_before_gen = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"System RAM before generation: {mem_before_gen:.2f}GB")
        if torch.cuda.is_available():
            gpu_before_gen = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"GPU memory before generation: {gpu_before_gen:.2f}GB")
        
        all_metadata = []
        
        logger.info("Starting generation loop...")
        logger.info(f"Dataset type: {type(ds)}")
        logger.info(f"Dataset format: {ds.format}")
        
        # Critical: Test accessing first sample BEFORE tqdm
        logger.info("Testing dataset access...")
        try:
            first_sample = ds[0]
            logger.info(f"First sample keys: {dict(first_sample).keys()}")
            logger.info(f"First sample messages type: {type(first_sample['messages'])}")
            logger.info(f"First sample audios type: {type(first_sample['audios'])}")
            del first_sample
            gc.collect()
            logger.info("✓ Dataset access test passed")
        except Exception as e:
            logger.error(f"Dataset access test FAILED: {e}")
            raise
        
        # Check memory after test access
        mem_after_test = psutil.Process().memory_info().rss / 1024**3
        gpu_after_test = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
        logger.info(f"After test access - RAM: {mem_after_test:.2f}GB, GPU: {gpu_after_test:.2f}GB")
        
        with tqdm(total=len(ds) * samples_per_reference, desc="Generating") as pbar:
            for ref_idx, ref_sample in enumerate(ds):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing reference {ref_idx}/{len(ds)}")
                
                # Log memory at start of each reference
                mem_ref_start = psutil.Process().memory_info().rss / 1024**3
                gpu_ref_start = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                logger.info(f"Ref {ref_idx} start - RAM: {mem_ref_start:.2f}GB, GPU: {gpu_ref_start:.2f}GB")
                
                sample_dict = dict(ref_sample)
                logger.info(f"Sample dict keys: {sample_dict.keys()}")
                messages = sample_dict["messages"]
                
                ref_text = None
                for msg in messages:
                    if msg["role"] == "assistant":
                        ref_text = msg["content"]
                        break
                
                if not ref_text:
                    logger.error(f"No ref text in sample {ref_idx}")
                    pbar.update(samples_per_reference)
                    continue
                
                audio_paths = sample_dict["audios"]
                logger.info(f"Audio paths type: {type(audio_paths)}, value: {audio_paths}")
                
                if not audio_paths:
                    logger.error(f"No audio in sample {ref_idx}")
                    pbar.update(samples_per_reference)
                    continue
                
                audio_path = Path(audio_paths[0])
                logger.info(f"Audio path: {audio_path}, exists: {audio_path.exists()}, size: {audio_path.stat().st_size / 1024**2:.2f}MB" if audio_path.exists() else f"Audio path: {audio_path}, exists: False")
                if not audio_path.exists():
                    logger.error(f"Audio not found: {audio_path}")
                    pbar.update(samples_per_reference)
                    continue
                
                try:
                    audio_array, audio_sr = sf.read(str(audio_path))
                    audio_array = np.array(audio_array, dtype=np.float32)
                    if audio_array.ndim > 1:
                        audio_array = audio_array.mean(axis=1)
                    logger.debug(f"Loaded audio: {audio_path.name}, shape={audio_array.shape}, sr={audio_sr}")
                except Exception as e:
                    logger.error(f"Failed to load audio: {e}")
                    pbar.update(samples_per_reference)
                    continue
                
                # Log memory before processing this reference
                if ref_idx == 0:
                    mem_first_ref = psutil.Process().memory_info().rss / 1024**3
                    logger.info(f"System RAM at first reference: {mem_first_ref:.2f}GB")
                    if torch.cuda.is_available():
                        gpu_first_ref = torch.cuda.memory_allocated(0) / 1024**3
                        logger.info(f"GPU memory at first reference: {gpu_first_ref:.2f}GB")
                
                for sample_idx in range(samples_per_reference):
                    # Lazy load random target text
                    text_idx = random.randint(0, num_target_texts - 1)
                    logger.debug(f"[Ref {ref_idx}, Sample {sample_idx}] Loading target text at index {text_idx}")
                    target_text = self._get_text_at_index(target_texts_file, text_idx)
                    logger.debug(f"[Ref {ref_idx}, Sample {sample_idx}] Target text ({len(target_text)} chars): {target_text[:50]}...")
                    
                    ref_folder = f"speaker_{start_idx + ref_idx:05d}"
                    ref_audio_dir = self.audio_dir / ref_folder
                    ref_audio_dir.mkdir(exist_ok=True)
                    
                    output_path = str(ref_audio_dir / f"text_id_{text_idx}.wav")
                    
                    try:
                        logger.debug(f"Generating sample {sample_idx+1}/{samples_per_reference} for ref {ref_idx}")
                        metadata = self.generate_sample(
                            ref_audio_data=audio_array,
                            ref_sample_rate=audio_sr,
                            ref_text=ref_text,
                            target_text=target_text,
                            target_text_idx=text_idx,
                            output_path=output_path,
                            speaker_id=f"speaker_{start_idx + ref_idx:05d}"
                        )
                        all_metadata.append(metadata)
                        logger.debug(f"✓ Sample generated successfully")
                    except Exception as e:
                        logger.error(f"Generation failed: {e}")
                        import traceback
                        logger.error(f"Full traceback:\n{traceback.format_exc()}")
                        raise  # Re-raise to see full error
                    
                    pbar.update(1)
                    
                    # Log memory every 10 samples
                    if len(all_metadata) % 10 == 0 and len(all_metadata) > 0:
                        mem_current = psutil.Process().memory_info().rss / 1024**3
                        logger.info(f"[Sample {len(all_metadata)}] System RAM: {mem_current:.2f}GB")
                        if torch.cuda.is_available():
                            gpu_current = torch.cuda.memory_allocated(0) / 1024**3
                            gpu_reserved_current = torch.cuda.memory_reserved(0) / 1024**3
                            logger.info(f"[Sample {len(all_metadata)}] GPU: {gpu_current:.2f}GB allocated, {gpu_reserved_current:.2f}GB reserved")
                
                if (ref_idx + 1) % 10 == 0:
                    self._save_metadata(all_metadata, f"checkpoint_{ref_idx+1}.json")
                    # Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    mem_checkpoint = psutil.Process().memory_info().rss / 1024**3
                    logger.info(f"[Checkpoint {ref_idx+1}] System RAM: {mem_checkpoint:.2f}GB, Samples: {len(all_metadata)}")
                    if torch.cuda.is_available():
                        gpu_checkpoint = torch.cuda.memory_allocated(0) / 1024**3
                        gpu_reserved_checkpoint = torch.cuda.memory_reserved(0) / 1024**3
                        logger.info(f"[Checkpoint {ref_idx+1}] GPU: {gpu_checkpoint:.2f}GB allocated, {gpu_reserved_checkpoint:.2f}GB reserved")
        
        logger.success(f"Generated {len(all_metadata)} samples")
        return all_metadata
    
    def _get_text_at_index(self, file_path: str, target_idx: int) -> str:
        """Lazy load a specific text line without loading entire file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i == target_idx:
                    return line
        raise IndexError(f"Text index {target_idx} not found")
    
    def _save_metadata(self, metadata_list: List[SyntheticSample], filename: str):
        """Save metadata to JSON."""
        with open(self.metadata_dir / filename, 'w', encoding='utf-8') as f:
            json.dump([asdict(m) for m in metadata_list], f, indent=2, ensure_ascii=False)
    
    def save_final_metadata(self, metadata_list: List[SyntheticSample]):
        """Save final metadata and statistics."""
        self._save_metadata(metadata_list, "metadata_complete.json")
        
        stats = {
            "total_samples": len(metadata_list),
            "total_duration_hours": round(sum(m.duration_sec for m in metadata_list) / 3600, 2),
            "unique_speakers": len(set(m.reference_speaker_id for m in metadata_list if m.reference_speaker_id)),
            "sample_rate": self.sample_rate,
            "engine": "vLLM-Client-v3"
        }
        
        with open(self.metadata_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("=" * 80)
        logger.info(f"✅ COMPLETE! {stats['total_samples']} samples, {stats['total_duration_hours']}h")
        logger.info(f"📁 {self.output_dir}")
        logger.info("=" * 80)


# Removed load_target_texts - using lazy loading instead


def main():
    parser = argparse.ArgumentParser(description="Synthetic TTS Data Generator v3")
    parser.add_argument("--vllm-server-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="audarai/auralix_flash_3")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--target-texts-file", required=True)
    parser.add_argument("--output-dir", default="data/tts/synthetic_v3")
    parser.add_argument("--samples-per-reference", type=int, default=10)
    parser.add_argument("--codec-repo", default="neuphonic/neucodec")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    try:
        generator = VLLMClientSyntheticGenerator(
            vllm_server_url=args.vllm_server_url,
            model_name=args.model_name,
            output_dir=args.output_dir,
            codec_repo=args.codec_repo,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens
        )
        
        metadata = generator.generate_from_dataset(
            dataset_path=args.dataset_path,
            target_texts_file=args.target_texts_file,
            samples_per_reference=args.samples_per_reference,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        
        generator.save_final_metadata(metadata)
        logger.success("✅ Done!")
        
    except Exception as e:
        logger.exception(f"Failed: {e}")
        raise


if __name__ == "__main__":
    main()
