import os
import re
import sys
import io
import logging
import time
import numpy as np
import torch
import gradio as gr
import soundfile as sf
from typing import Generator, Optional, Tuple
from funasr import AutoModel
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import voxcpm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------- Inline i18n (en + zh-CN only) ----------

_USAGE_INSTRUCTIONS_EN = (
    "**VoxCPM2 — Three Modes of Speech Generation:**\n\n"
    "🎨 **Voice Design** — Create a brand-new voice  \n"
    "No reference audio required. Describe the desired voice characteristics "
    "(gender, age, tone, emotion, pace …) in **Control Instruction**, and VoxCPM2 "
    "will craft a unique voice from your description alone.\n\n"
    "🎛️ **Controllable Cloning** — Clone a voice with optional style guidance  \n"
    "Upload a reference audio clip, then use **Control Instruction** to steer "
    "emotion, speaking pace, and overall style while preserving the original timbre.\n\n"
    "🎙️ **Ultimate Cloning** — Reproduce every vocal nuance through audio continuation  \n"
    "Turn on **Ultimate Cloning Mode** and provide (or auto-transcribe) the reference audio's transcript. "
    "The model treats the reference clip as a spoken prefix and seamlessly **continues** from it, faithfully preserving every vocal detail."
    "Note: This mode will disable Control Instruction."
)

_EXAMPLES_FOOTER_EN = (
    "---\n"
    "**💡 Voice Description Examples:**  \n"
    "Try the following Control Instructions to explore different voices:  \n\n"
    "**Example 1 — Gentle & Melancholic Girl**  \n"
    '`Control Instruction`: *"A young girl with a soft, sweet voice. '
    'Speaks slowly with a melancholic, slightly tsundere tone."*  \n'
    '`Target Text`: *"I never asked you to stay… It\'s not like I care or anything. '
    'But… why does it still hurt so much now that you\'re gone?"*  \n\n'
    "**Example 2 — Laid-Back Surfer Dude**  \n"
    '`Control Instruction`: *"Relaxed young male voice, slightly nasal, '
    'lazy drawl, very casual and chill."*  \n'
    '`Target Text`: *"Dude, did you see that set? The waves out there are totally gnarly today. '
    "Just catching barrels all morning — it's like, totally righteous, you know what I mean?\"*"
)

_USAGE_INSTRUCTIONS_ZH = (
    "**VoxCPM2 — 三种语音生成方式：**\n\n"
    "🎨 **声音设计（Voice Design）**  \n"
    "无需参考音频。在 **Control Instruction** 中描述目标音色特征"
    "（性别、年龄、语气、情绪、语速等），VoxCPM2 即可为你从零创造独一无二的声音。\n\n"
    "🎛️ **可控克隆（Controllable Cloning）**  \n"
    "上传参考音频，同时可选地使用 **Control Instruction** 来指定情绪、语速、风格等表达方式，"
    "在保留原始音色的基础上灵活控制说话风格。\n\n"
    "🎙️ **极致克隆（Ultimate Cloning）**  \n"
    "开启 **极致克隆模式** 并提供参考音频的文字内容（可自动识别）。"
    "模型会将参考音频视为已说出的前文，以**音频续写**的方式完整还原参考音频中的所有声音细节。"
    "注意：该模式与可控克隆模式互斥，将禁用Control Instruction。\n\n"
)

_EXAMPLES_FOOTER_ZH = (
    "---\n"
    "**💡 声音描述示例（中英文均可）：**  \n\n"
    "**示例 1 — 深宫太后**  \n"
    '`Control Instruction`: *"中老年女性，声音低沉阴冷，语速缓慢而有力，'
    '字字深思熟虑，带有深不可测的城府与威慑感。"*  \n'
    '`Target Text`: *"哀家在这深宫待了四十年，什么风浪没见过？你以为瞒得过哀家？"*  \n\n'
    "**示例 2 — 暴躁驾校教练**  \n"
    '`Control Instruction`: *"暴躁的中年男声，语速快，充满无奈和愤怒"*  \n'
    '`Target Text`: *"踩离合！踩刹车啊！你往哪儿开呢？前面是树你看不见吗？'
    '我教了你八百遍了，打死方向盘！你是不是想把车给我开到沟里去？"*  \n\n'
    "---\n"
    "**🗣️ 方言生成指南：**  \n"
    "要生成地道的方言语音，请在 **Target Text** 中直接使用方言词汇和句式，"
    "并在 **Control Instruction** 中描述方言特征。  \n\n"
    "**示例 — 广东话**  \n"
    '`Control Instruction`: *"粤语，中年男性，语气平淡"*  \n'
    '✅ 正确（粤语表达）：*"伙計，唔該一個A餐，凍奶茶少甜！"*  \n'
    '❌ 错误（普通话原文）：*"伙计，麻烦来一个A餐，冻奶茶少甜！"*  \n\n'
    "**示例 — 河南话**  \n"
    '`Control Instruction`: *"河南话，接地气的大叔"*  \n'
    '✅ 正确（河南话表达）：*"恁这是弄啥嘞？晌午吃啥饭？"*  \n'
    '❌ 错误（普通话原文）：*"你这是在干什么呢？中午吃什么饭？"*  \n\n'
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到 Target Text 中即可。  \n\n"
)

_I18N_TRANSLATIONS = {
    "en": {
        "reference_audio_label": "🎤 Reference Audio (optional — upload for cloning)",
        "prompt_audio_label": "🎙️ Prompt Audio (optional — for audio continuation)",
        "show_prompt_text_label": "🎙️ Ultimate Cloning Mode (transcript-guided cloning)",
        "show_prompt_text_info": "Auto-transcribes reference audio for every vocal nuance reproduced. Control Instruction will be disabled when active.",
        "prompt_text_label": "Transcript of Reference Audio (auto-filled via ASR, editable)",
        "prompt_text_placeholder": "The transcript of your reference audio will appear here …",
        "control_label": "🎛️ Control Instruction (optional — supports Chinese & English)",
        "control_placeholder": "e.g. A warm young woman / 年轻女性，温柔甜美 / Excited and fast-paced",
        "target_text_label": "✍️ Target Text — the content to speak",
        "generate_btn": "🔊 Generate Speech",
        "generated_audio_label": "Generated Audio",
        "first_token_latency_label": "First token latency",
        "total_generation_latency_label": "Total generation time",
        "advanced_settings_title": "⚙️ Advanced Settings",
        "ref_denoise_label": "Reference audio enhancement",
        "ref_denoise_info": "Apply ZipEnhancer denoising to the reference audio before cloning",
        "normalize_label": "Text normalization",
        "normalize_info": "Normalize numbers, dates, and abbreviations via wetext",
        "cfg_label": "CFG (guidance scale)",
        "cfg_info": "Higher → closer to the prompt / reference; lower → more creative variation",
        "dit_steps_label": "LocDiT flow-matching steps",
        "dit_steps_info": "LocDiT flow-matching steps — more steps → maybe better audio quality, but slower",
        "min_len_label": "Minimum generated audio tokens",
        "max_len_label": "Maximum generated audio tokens",
        "retry_badcase_label": "Retry bad cases",
        "retry_badcase_max_times_label": "Maximum retry attempts",
        "retry_badcase_ratio_threshold_label": "Bad-case audio/text ratio threshold",
        "trim_silence_vad_label": "Trim prompt/reference silence with VAD",
        "streaming_prefix_len_label": "Streaming decode prefix length",
        "usage_instructions": _USAGE_INSTRUCTIONS_EN,
        "examples_footer": _EXAMPLES_FOOTER_EN,
    },
    "zh-CN": {
        "reference_audio_label": "🎤 参考音频（可选 — 上传后用于克隆）",
        "prompt_audio_label": "🎙️ Prompt Audio（可选 — 用于音频续写）",
        "show_prompt_text_label": "🎙️ 极致克隆模式（基于文本引导的极致克隆）",
        "show_prompt_text_info": "自动识别参考音频文本，完整还原音色、节奏、情感等全部声音细节。开启后 Control Instruction 将暂时禁用",
        "prompt_text_label": "参考音频内容文本（ASR 自动填充，可手动编辑）",
        "prompt_text_placeholder": "参考音频的文字内容将自动识别并显示在此处 …",
        "control_label": "🎛️ Control Instruction（可选 — 支持中英文描述）",
        "control_placeholder": "如：年轻女性，温柔甜美 / A warm young woman / 暴躁老哥，语速飞快",
        "target_text_label": "✍️ Target Text — 要合成的目标文本",
        "generate_btn": "🔊 开始生成",
        "generated_audio_label": "生成结果",
        "first_token_latency_label": "首个音频 token 延迟",
        "total_generation_latency_label": "总生成耗时",
        "advanced_settings_title": "⚙️ 高级设置",
        "ref_denoise_label": "参考音频降噪增强",
        "ref_denoise_info": "克隆前使用 ZipEnhancer 对参考音频进行降噪处理",
        "normalize_label": "文本规范化",
        "normalize_info": "自动规范化数字、日期及缩写（基于 wetext）",
        "cfg_label": "CFG（引导强度）",
        "cfg_info": "数值越高 → 越贴合提示/参考音色；数值越低 → 生成风格更自由",
        "dit_steps_label": "LocDiT 流匹配迭代步数",
        "dit_steps_info": "LocDiT 流匹配生成迭代步数 — 步数越多 → 可能生成更好的音频质量，但速度变慢",
        "min_len_label": "最小生成音频 token 数",
        "max_len_label": "最大生成音频 token 数",
        "retry_badcase_label": "重试异常生成结果",
        "retry_badcase_max_times_label": "最大重试次数",
        "retry_badcase_ratio_threshold_label": "异常音频/文本比例阈值",
        "trim_silence_vad_label": "使用 VAD 裁剪提示/参考音频静音",
        "streaming_prefix_len_label": "流式解码前缀长度",
        "usage_instructions": _USAGE_INSTRUCTIONS_ZH,
        "examples_footer": _EXAMPLES_FOOTER_ZH,
    },
    "zh-Hans": None,  # alias, filled below
    "zh": None,       # alias, filled below
}
_I18N_TRANSLATIONS["zh-Hans"] = _I18N_TRANSLATIONS["zh-CN"]
_I18N_TRANSLATIONS["zh"] = _I18N_TRANSLATIONS["zh-CN"]

for _d in _I18N_TRANSLATIONS.values():
    if _d is not None:
        for _k, _v in _I18N_TRANSLATIONS["en"].items():
            _d.setdefault(_k, _v)

I18N = gr.I18n(**_I18N_TRANSLATIONS)

DEFAULT_TARGET_TEXT = (
    "VoxCPM2 is a creative multilingual TTS model from ModelBest, "
    "designed to generate highly realistic speech."
)

_CUSTOM_CSS = """
.logo-container {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
}
.logo-container img {
    height: 80px;
    width: auto;
    max-width: 200px;
    display: inline-block;
}

/* Toggle switch style */
.switch-toggle {
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--block-background-fill);
}
.switch-toggle input[type="checkbox"] {
    appearance: none;
    -webkit-appearance: none;
    width: 44px;
    height: 24px;
    background: #ccc;
    border-radius: 12px;
    position: relative;
    cursor: pointer;
    transition: background 0.3s ease;
    flex-shrink: 0;
}
.switch-toggle input[type="checkbox"]::after {
    content: "";
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    background: white;
    border-radius: 50%;
    transition: transform 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.switch-toggle input[type="checkbox"]:checked {
    background: var(--color-accent);
}
.switch-toggle input[type="checkbox"]:checked::after {
    transform: translateX(20px);
}
"""

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
)


# ---------- Model ----------

class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Running on device: {self.device}")
        self._model_id = model_id

        self.asr_model_id = "iic/SenseVoiceSmall"
        logger.info(f"Loading ASR model: {self.asr_model_id}")
        self.asr_model: Optional[AutoModel] = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device="cuda:0" if self.device == "cuda" else "cpu",
        )
        logger.info("ASR model loaded successfully.")

        self.voxcpm_model: voxcpm.VoxCPM = self._load_voxcpm_model()

    def _load_voxcpm_model(self) -> voxcpm.VoxCPM:
        logger.info(f"Loading model: {self._model_id}")
        model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=True,
            device=self.device,
        )
        if self.device == "cuda":
            torch.cuda.synchronize()
        logger.info("Model loaded successfully.")
        return model

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        return self.voxcpm_model

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
        self,
        *,
        final_text: str,
        reference_audio_path: Optional[str],
        prompt_audio_path: Optional[str],
        prompt_text_clean: Optional[str],
        cfg_value_input: float,
        do_normalize: bool,
        denoise: bool,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        trim_silence_vad: bool = False,
        streaming_prefix_len: int = 4,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=reference_audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            min_len=min_len,
            max_len=max_len,
            normalize=do_normalize,
            denoise=denoise,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            trim_silence_vad=trim_silence_vad,
            streaming_prefix_len=streaming_prefix_len,
        )
        if prompt_text_clean and prompt_audio_path:
            generate_kwargs["prompt_wav_path"] = prompt_audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def _prepare_generate_kwargs(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        trim_silence_vad: bool = False,
        streaming_prefix_len: int = 4,
    ) -> tuple[voxcpm.VoxCPM, dict]:
        current_model = self.get_or_load_voxcpm()

        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("Please input text to synthesize.")

        control = (control_instruction or "").strip()
        # Strip any parentheses (half-width/full-width) from control text to avoid
        # breaking the "(control)text" prompt format expected by the model.
        control = re.sub(r"[()（）]", "", control).strip()
        final_text = f"({control}){text}" if control else text

        reference_audio_path = reference_wav_path_input if reference_wav_path_input else None
        prompt_text_clean = (prompt_text or "").strip() or None
        prompt_audio_path = prompt_wav_path_input if prompt_wav_path_input else None
        if prompt_text_clean and prompt_audio_path is None and reference_audio_path:
            prompt_audio_path = reference_audio_path

        if reference_audio_path and prompt_audio_path and prompt_text_clean:
            logger.info(f"[Voice Cloning] prompt_wav + prompt_text + reference_wav")
        elif prompt_audio_path and prompt_text_clean:
            logger.info(f"[Audio Continuation] prompt_wav + prompt_text")
        elif reference_audio_path:
            logger.info(f"[Voice Control] reference_wav only")
        else:
            logger.info(f"[Voice Design] control: {control[:50] if control else 'None'}...")

        logger.info(f"Generating audio for text: '{final_text[:80]}...'")
        generate_kwargs = self._build_generate_kwargs(
            final_text=final_text,
            reference_audio_path=reference_audio_path,
            prompt_audio_path=prompt_audio_path,
            prompt_text_clean=prompt_text_clean,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            min_len=min_len,
            max_len=max_len,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            trim_silence_vad=trim_silence_vad,
            streaming_prefix_len=streaming_prefix_len,
        )
        return current_model, generate_kwargs

    def generate_tts_audio(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        trim_silence_vad: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Tuple[int, np.ndarray]:
        current_model, generate_kwargs = self._prepare_generate_kwargs(
            text_input=text_input,
            control_instruction=control_instruction,
            reference_wav_path_input=reference_wav_path_input,
            prompt_wav_path_input=prompt_wav_path_input,
            prompt_text=prompt_text,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            min_len=min_len,
            max_len=max_len,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            trim_silence_vad=trim_silence_vad,
            streaming_prefix_len=streaming_prefix_len,
        )
        wav = current_model.generate(**generate_kwargs)
        return (current_model.tts_model.sample_rate, wav)

    def generate_tts_audio_streaming(
        self,
        text_input: str,
        control_instruction: str = "",
        reference_wav_path_input: Optional[str] = None,
        prompt_wav_path_input: Optional[str] = None,
        prompt_text: str = "",
        cfg_value_input: float = 2.0,
        do_normalize: bool = True,
        denoise: bool = True,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        trim_silence_vad: bool = False,
        streaming_prefix_len: int = 4,
    ) -> Generator[tuple[bytes, Optional[float], float], None, None]:
        current_model, generate_kwargs = self._prepare_generate_kwargs(
            text_input=text_input,
            control_instruction=control_instruction,
            reference_wav_path_input=reference_wav_path_input,
            prompt_wav_path_input=prompt_wav_path_input,
            prompt_text=prompt_text,
            cfg_value_input=cfg_value_input,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=inference_timesteps,
            min_len=min_len,
            max_len=max_len,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            trim_silence_vad=trim_silence_vad,
            streaming_prefix_len=streaming_prefix_len,
        )
        start_time = time.perf_counter()
        first_token_latency = None
        for chunk in current_model.generate_streaming(**generate_kwargs):
            if first_token_latency is None:
                first_token_latency = time.perf_counter() - start_time
            yield (
                self._encode_wav_chunk(current_model.tts_model.sample_rate, chunk),
                first_token_latency,
                time.perf_counter() - start_time,
            )

    @staticmethod
    def _encode_wav_chunk(sample_rate: int, wav: np.ndarray) -> bytes:
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, wav, sample_rate, format="WAV", subtype="PCM_16")
        return wav_buffer.getvalue()


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):
    gr.set_static_paths(paths=[Path.cwd().absolute() / "assets"])

    def _generate(
        text: str,
        control_instruction: str,
        ref_wav: Optional[str],
        prompt_wav: Optional[str],
        use_prompt_text: bool,
        prompt_text_value: str,
        cfg_value: float,
        do_normalize: bool,
        denoise: bool,
        dit_steps: int,
        min_len: int,
        max_len: int,
        retry_badcase: bool,
        retry_badcase_max_times: int,
        retry_badcase_ratio_threshold: float,
        trim_silence_vad: bool,
        streaming_prefix_len: int,
    ):
        actual_prompt_text = prompt_text_value.strip() if use_prompt_text else ""
        actual_control = "" if use_prompt_text else control_instruction
        yield None, "Waiting for first audio chunk...", "Generating..."
        last_first_latency = None
        last_total_latency = None
        for wav_chunk, first_latency, total_latency in demo.generate_tts_audio_streaming(
            text_input=text,
            control_instruction=actual_control,
            reference_wav_path_input=ref_wav,
            prompt_wav_path_input=prompt_wav,
            prompt_text=actual_prompt_text,
            cfg_value_input=cfg_value,
            do_normalize=do_normalize,
            denoise=denoise,
            inference_timesteps=int(dit_steps),
            min_len=int(min_len),
            max_len=int(max_len),
            retry_badcase=retry_badcase,
            retry_badcase_max_times=int(retry_badcase_max_times),
            retry_badcase_ratio_threshold=float(retry_badcase_ratio_threshold),
            trim_silence_vad=trim_silence_vad,
            streaming_prefix_len=int(streaming_prefix_len),
        ):
            last_first_latency = first_latency
            last_total_latency = total_latency
            yield (
                wav_chunk,
                f"{first_latency:.3f} s",
                f"{total_latency:.3f} s",
            )

        if last_first_latency is None or last_total_latency is None:
            yield None, "No audio generated", "0.000 s"

    def _on_toggle_instant(checked):
        """Instant UI toggle — no ASR, no blocking."""
        if checked:
            return (
                gr.update(visible=True, value="", placeholder="Recognizing reference audio..."),
                gr.update(visible=False),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True, interactive=True),
        )

    def _run_asr_if_needed(checked, reference_audio_path, prompt_audio_path):
        """Run ASR after the UI has updated. Only when toggled ON."""
        audio_path = prompt_audio_path or reference_audio_path
        if not checked or not audio_path:
            return gr.update()
        try:
            logger.info("Running ASR on reference audio...")
            asr_text = demo.prompt_wav_recognition(audio_path)
            logger.info(f"ASR result: {asr_text[:60]}...")
            return gr.update(value=asr_text)
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
            return gr.update(value="")

    with gr.Blocks() as interface:
        gr.HTML(
            '<div class="logo-container">'
            '<img src="/gradio_api/file=assets/voxcpm_logo.png" alt="VoxCPM Logo">'
            "</div>"
        )

        gr.Markdown(I18N("usage_instructions"))

        with gr.Row():
            with gr.Column():
                reference_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("reference_audio_label"),
                )
                prompt_wav = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label=I18N("prompt_audio_label"),
                )
                show_prompt_text = gr.Checkbox(
                    value=False,
                    label=I18N("show_prompt_text_label"),
                    info=I18N("show_prompt_text_info"),
                    elem_classes=["switch-toggle"],
                )
                prompt_text = gr.Textbox(
                    value="",
                    label=I18N("prompt_text_label"),
                    placeholder=I18N("prompt_text_placeholder"),
                    lines=2,
                    visible=False,
                )
                control_instruction = gr.Textbox(
                    value="",
                    label=I18N("control_label"),
                    placeholder=I18N("control_placeholder"),
                    lines=2,
                )
                text = gr.Textbox(
                    value=DEFAULT_TARGET_TEXT,
                    label=I18N("target_text_label"),
                    lines=3,
                )

                with gr.Accordion(I18N("advanced_settings_title"), open=False):
                    DoDenoisePromptAudio = gr.Checkbox(
                        value=False,
                        label=I18N("ref_denoise_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("ref_denoise_info"),
                    )
                    DoNormalizeText = gr.Checkbox(
                        value=False,
                        label=I18N("normalize_label"),
                        elem_classes=["switch-toggle"],
                        info=I18N("normalize_info"),
                    )
                    cfg_value = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=2.0,
                        step=0.1,
                        label=I18N("cfg_label"),
                        info=I18N("cfg_info"),
                    )
                    dit_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label=I18N("dit_steps_label"),
                        info=I18N("dit_steps_info"),
                    )
                    min_len = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=2,
                        step=1,
                        label=I18N("min_len_label"),
                    )
                    max_len = gr.Slider(
                        minimum=16,
                        maximum=8192,
                        value=4096,
                        step=16,
                        label=I18N("max_len_label"),
                    )
                    retry_badcase = gr.Checkbox(
                        value=True,
                        label=I18N("retry_badcase_label"),
                        elem_classes=["switch-toggle"],
                    )
                    retry_badcase_max_times = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label=I18N("retry_badcase_max_times_label"),
                    )
                    retry_badcase_ratio_threshold = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=6.0,
                        step=0.5,
                        label=I18N("retry_badcase_ratio_threshold_label"),
                    )
                    trim_silence_vad = gr.Checkbox(
                        value=False,
                        label=I18N("trim_silence_vad_label"),
                        elem_classes=["switch-toggle"],
                    )
                    streaming_prefix_len = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        label=I18N("streaming_prefix_len_label"),
                    )

                run_btn = gr.Button(I18N("generate_btn"), variant="primary", size="lg")

            with gr.Column():
                audio_output = gr.Audio(label=I18N("generated_audio_label"), streaming=True, autoplay=True)
                first_token_latency = gr.Textbox(
                    value="",
                    label=I18N("first_token_latency_label"),
                    interactive=False,
                )
                total_generation_latency = gr.Textbox(
                    value="",
                    label=I18N("total_generation_latency_label"),
                    interactive=False,
                )
                gr.Markdown(I18N("examples_footer"))

        show_prompt_text.change(
            fn=_on_toggle_instant,
            inputs=[show_prompt_text],
            outputs=[prompt_text, control_instruction],
        ).then(
            fn=_run_asr_if_needed,
            inputs=[show_prompt_text, reference_wav, prompt_wav],
            outputs=[prompt_text],
        )

        run_btn.click(
            fn=_generate,
            inputs=[
                text,
                control_instruction,
                reference_wav,
                prompt_wav,
                show_prompt_text,
                prompt_text,
                cfg_value,
                DoNormalizeText,
                DoDenoisePromptAudio,
                dit_steps,
                min_len,
                max_len,
                retry_badcase,
                retry_badcase_max_times,
                retry_badcase_ratio_threshold,
                trim_silence_vad,
                streaming_prefix_len,
            ],
            outputs=[audio_output, first_token_latency, total_generation_latency],
            show_progress=True,
            api_name="generate",
        )

    return interface

def run_demo(
    server_name: str = "0.0.0.0",
    server_port: int = 8808,
    show_error: bool = True,
    model_id: str = "openbmb/VoxCPM2",
    share: bool = True,
):
    demo = VoxCPMDemo(model_id=model_id)
    interface = create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=show_error,
        share=share,
        i18n=I18N,
        theme=_APP_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, default="openbmb/VoxCPM2",
        help="Local path or HuggingFace repo ID (default: openbmb/VoxCPM2)",
    )
    parser.add_argument("--port", type=int, default=8808, help="Server port")
    parser.add_argument("--no-share", action="store_true", help="Disable Gradio share link")
    args = parser.parse_args()
    run_demo(model_id=args.model_id, server_port=args.port, share=not args.no_share)
