#!/usr/bin/env python3
"""
Audio Validation Script
Verifies that generated audio files are playable and contain actual music.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import wave
import struct

def validate_wav_file(filepath):
    """Validate a WAV file and extract characteristics."""
    validation = {
        'file': str(filepath),
        'exists': False,
        'readable': False,
        'valid_wav': False,
        'playable': False,
        'contains_audio': False,
        'characteristics': {},
        'issues': []
    }
    
    # Check if file exists
    if not filepath.exists():
        validation['issues'].append("File does not exist")
        return validation
    
    validation['exists'] = True
    
    # Check file size
    file_size = filepath.stat().st_size
    if file_size < 44:  # WAV header is at least 44 bytes
        validation['issues'].append("File too small to be valid WAV")
        return validation
    
    validation['characteristics']['file_size_kb'] = file_size / 1024
    
    try:
        # Try to open as WAV file
        with wave.open(str(filepath), 'rb') as wav:
            validation['readable'] = True
            
            # Get WAV parameters
            params = wav.getparams()
            validation['characteristics']['channels'] = params.nchannels
            validation['characteristics']['sample_rate'] = params.framerate
            validation['characteristics']['sample_width'] = params.sampwidth
            validation['characteristics']['frames'] = params.nframes
            validation['characteristics']['duration_seconds'] = params.nframes / params.framerate
            
            # Validate parameters
            if params.framerate not in [16000, 22050, 24000, 32000, 44100, 48000]:
                validation['issues'].append(f"Unusual sample rate: {params.framerate}")
            
            if params.nchannels not in [1, 2]:
                validation['issues'].append(f"Unusual channel count: {params.nchannels}")
            
            if params.sampwidth not in [1, 2, 3, 4]:
                validation['issues'].append(f"Unusual sample width: {params.sampwidth}")
            
            validation['valid_wav'] = True
            
            # Read audio data
            frames = wav.readframes(min(params.framerate * 2, params.nframes))  # Read up to 2 seconds
            
            # Convert to numpy array
            if params.sampwidth == 2:  # 16-bit
                audio = np.frombuffer(frames, dtype=np.int16)
            elif params.sampwidth == 4:  # 32-bit
                audio = np.frombuffer(frames, dtype=np.int32)
            else:
                audio = np.frombuffer(frames, dtype=np.uint8)
            
            # Normalize to [-1, 1]
            if params.sampwidth == 2:
                audio = audio.astype(np.float32) / 32768.0
            elif params.sampwidth == 4:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = (audio.astype(np.float32) - 128) / 128.0
            
            # Audio analysis
            if len(audio) > 0:
                validation['characteristics']['rms'] = float(np.sqrt(np.mean(audio**2)))
                validation['characteristics']['peak'] = float(np.abs(audio).max())
                validation['characteristics']['mean'] = float(np.mean(audio))
                validation['characteristics']['std'] = float(np.std(audio))
                
                # Check if contains actual audio (not silence)
                if validation['characteristics']['rms'] > 0.001:
                    validation['contains_audio'] = True
                    validation['playable'] = True
                else:
                    validation['issues'].append("Audio is silent or near-silent")
                
                # Check for clipping
                if validation['characteristics']['peak'] > 0.99:
                    validation['issues'].append("Audio may be clipping")
                
                # Frequency analysis (simple)
                fft = np.fft.fft(audio[:8192] if len(audio) > 8192 else audio)
                freqs = np.fft.fftfreq(len(fft), 1/params.framerate)
                magnitude = np.abs(fft)
                
                # Find dominant frequency
                idx = np.argmax(magnitude[:len(magnitude)//2])
                dominant_freq = abs(freqs[idx])
                validation['characteristics']['dominant_frequency'] = float(dominant_freq)
                
                # Check frequency content
                freq_energy = np.sum(magnitude[1:len(magnitude)//2])  # Exclude DC
                if freq_energy < 0.1:
                    validation['issues'].append("Very low frequency content")
            
    except Exception as e:
        validation['issues'].append(f"Error reading file: {str(e)}")
        validation['valid_wav'] = False
    
    return validation


def validate_directory(directory):
    """Validate all WAV files in a directory."""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return
    
    wav_files = list(directory.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {directory}")
        return
    
    print(f"🎵 Validating {len(wav_files)} audio files in {directory}")
    print("="*60)
    
    results = []
    valid_count = 0
    playable_count = 0
    
    for wav_file in wav_files:
        print(f"\nValidating: {wav_file.name}")
        
        validation = validate_wav_file(wav_file)
        results.append(validation)
        
        if validation['valid_wav']:
            valid_count += 1
            print(f"  ✓ Valid WAV file")
            
            chars = validation['characteristics']
            print(f"  Duration: {chars.get('duration_seconds', 0):.1f}s")
            print(f"  Sample rate: {chars.get('sample_rate', 0)} Hz")
            print(f"  RMS level: {chars.get('rms', 0):.3f}")
            
            if validation['playable']:
                playable_count += 1
                print(f"  ✓ Contains playable audio")
            else:
                print(f"  ⚠️  May not be playable")
            
            if validation['issues']:
                print(f"  Issues: {', '.join(validation['issues'])}")
        else:
            print(f"  ❌ Invalid WAV file")
            if validation['issues']:
                for issue in validation['issues']:
                    print(f"    - {issue}")
    
    # Summary report
    print("\n" + "="*60)
    print("📊 VALIDATION SUMMARY")
    print("="*60)
    print(f"Total files: {len(wav_files)}")
    print(f"Valid WAV files: {valid_count} ({valid_count/len(wav_files)*100:.0f}%)")
    print(f"Playable audio: {playable_count} ({playable_count/len(wav_files)*100:.0f}%)")
    
    # Detailed statistics
    if playable_count > 0:
        print("\nPlayable Audio Files:")
        durations = []
        rms_levels = []
        
        for r in results:
            if r['playable']:
                name = Path(r['file']).name
                dur = r['characteristics'].get('duration_seconds', 0)
                rms = r['characteristics'].get('rms', 0)
                
                durations.append(dur)
                rms_levels.append(rms)
                
                print(f"  ✓ {name} - {dur:.1f}s, RMS: {rms:.3f}")
        
        print(f"\nAggregate Statistics:")
        print(f"  Total duration: {sum(durations):.1f} seconds")
        print(f"  Average duration: {np.mean(durations):.1f} seconds")
        print(f"  Average RMS level: {np.mean(rms_levels):.3f}")
    
    # Save validation report
    report = {
        'directory': str(directory),
        'total_files': len(wav_files),
        'valid_files': valid_count,
        'playable_files': playable_count,
        'results': results
    }
    
    report_file = directory / "validation_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Validation report saved to: {report_file}")
    
    if playable_count > 0:
        print(f"\n✅ SUCCESS! {playable_count} files contain playable audio")
        print("These files can be played in any standard audio player.")
    else:
        print("\n❌ No playable audio files found")


def test_audio_playback():
    """Test if audio can be played on this system."""
    print("\n🔊 Testing Audio Playback Capability")
    print("-"*40)
    
    # Check for audio playback libraries
    audio_libs = {
        'pyaudio': False,
        'pygame': False,
        'playsound': False,
        'simpleaudio': False
    }
    
    for lib in audio_libs:
        try:
            __import__(lib)
            audio_libs[lib] = True
            print(f"✓ {lib} available")
        except ImportError:
            print(f"✗ {lib} not installed")
    
    if any(audio_libs.values()):
        print("\n✓ Audio playback libraries available")
        print("You can play the generated files using:")
        
        if audio_libs['pygame']:
            print("\n  With pygame:")
            print("  >>> import pygame")
            print("  >>> pygame.mixer.init()")
            print("  >>> pygame.mixer.music.load('file.wav')")
            print("  >>> pygame.mixer.music.play()")
        
        if audio_libs['playsound']:
            print("\n  With playsound:")
            print("  >>> from playsound import playsound")
            print("  >>> playsound('file.wav')")
    else:
        print("\n⚠️  No audio playback libraries installed")
        print("Install one with: pip install pygame")
    
    print("\n💡 You can also play the files with:")
    print("  - Any media player (VLC, Windows Media Player, etc.)")
    print("  - Web browser (drag and drop the file)")
    print("  - Audio editing software (Audacity, etc.)")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate MusicGen audio outputs')
    parser.add_argument('directory', nargs='?', default='test_outputs',
                        help='Directory containing audio files (default: test_outputs)')
    parser.add_argument('--test-playback', action='store_true',
                        help='Test audio playback capability')
    
    args = parser.parse_args()
    
    if args.test_playback:
        test_audio_playback()
    
    # Check common output directories
    directories_to_check = [
        args.directory,
        'test_outputs',
        'batch_outputs',
        'musicgen_outputs',
        '.'
    ]
    
    validated = False
    for dir_path in directories_to_check:
        if Path(dir_path).exists() and list(Path(dir_path).glob("*.wav")):
            validate_directory(dir_path)
            validated = True
            break
    
    if not validated:
        print("No directories with WAV files found.")
        print("Run one of the generation scripts first:")
        print("  - python simple_musicgen_test.py")
        print("  - python robust_musicgen_test.py")
        print("  - python batch_musicgen_test.py")


if __name__ == "__main__":
    main()