"""Transcription, context detection, model management, and streaming."""

from localwispr.transcribe.transcriber import (
    TranscriptionResult,
    WhisperTranscriber,
    transcribe_recording,
    transcribe_with_context,
)
from localwispr.transcribe.context import ContextDetector, ContextType
from localwispr.transcribe.streaming import (
    AudioBuffer,
    StreamingConfig,
    StreamingResult,
    StreamingTranscriber,
    SpeechSegment,
    VADProcessor,
    get_streaming_config,
)
from localwispr.transcribe.model_manager import (
    GGML_MODELS,
    MODEL_SIZES_MB,
    delete_model,
    download_model,
    get_all_models_status,
    get_available_model_names,
    get_model_disk_size,
    get_model_download_url,
    get_model_filename,
    get_model_path,
    get_model_status,
    get_models_dir,
    get_recommended_model_for_cpu,
    is_model_downloaded,
)
from localwispr.transcribe.download_progress import (
    DownloadProgressCallback,
    download_model_with_progress,
)
from localwispr.transcribe.device import (
    get_device_info,
    get_optimal_threads,
    resolve_device,
)
from localwispr.transcribe.gpu import (
    check_cuda_available,
    check_gpu,
    get_gpu_info,
    print_gpu_status,
    verify_whisper_gpu,
)

__all__ = [
    # transcriber
    "TranscriptionResult",
    "WhisperTranscriber",
    "transcribe_recording",
    "transcribe_with_context",
    # context
    "ContextDetector",
    "ContextType",
    # streaming
    "AudioBuffer",
    "StreamingConfig",
    "StreamingResult",
    "StreamingTranscriber",
    "SpeechSegment",
    "VADProcessor",
    "get_streaming_config",
    # model_manager
    "GGML_MODELS",
    "MODEL_SIZES_MB",
    "delete_model",
    "download_model",
    "get_all_models_status",
    "get_available_model_names",
    "get_model_disk_size",
    "get_model_download_url",
    "get_model_filename",
    "get_model_path",
    "get_model_status",
    "get_models_dir",
    "get_recommended_model_for_cpu",
    "is_model_downloaded",
    # download_progress
    "DownloadProgressCallback",
    "download_model_with_progress",
    # device
    "get_device_info",
    "get_optimal_threads",
    "resolve_device",
    # gpu
    "check_cuda_available",
    "check_gpu",
    "get_gpu_info",
    "print_gpu_status",
    "verify_whisper_gpu",
]
