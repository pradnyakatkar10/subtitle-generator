import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from typing import List, Tuple
import os
from typing import Iterator, TextIO, List
from gooey import Gooey, GooeyParser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LANG_CODE_MAPPER = {
    "en": ["english", "en_XX"],
    "zh": ["chinese", "zh_CN"],
    "de": ["german", "de_DE"],
    "es": ["spanish", "es_XX"],
    "ru": ["russian", "ru_RU"],
    "ko": ["korean", "ko_KR"],
    "fr": ["french", "fr_XX"],
    "ja": ["japanese", "ja_XX"],
    "pt": ["portuguese", "pt_XX"],
    "tr": ["turkish", "tr_TR"],
    "pl": ["polish", "pl_PL"],
    "nl": ["dutch", "nl_XX"],
    "ar": ["arabic", "ar_AR"],
    "sv": ["swedish", "sv_SE"],
    "it": ["italian", "it_IT"],
    "id": ["indonesian", "id_ID"],
    "hi": ["hindi", "hi_IN"],
    "fi": ["finnish", "fi_FI"],
    "vi": ["vietnamese", "vi_VN"],
    "he": ["hebrew", "he_IL"],
    "uk": ["ukrainian", "uk_UA"],
    "cs": ["czech", "cs_CZ"],
    "ro": ["romanian", "ro_RO"],
    "ta": ["tamil", "ta_IN"],
    "no": ["norwegian", ""],
    "th": ["thai", "th_TH"],
    "ur": ["urdu", "ur_PK"],
    "hr": ["croatian", "hr_HR"],
    "lt": ["lithuanian", "lt_LT"],
    "ml": ["malayalam", "ml_IN"],
    "te": ["telugu", "te_IN"],
    "fa": ["persian", "fa_IR"],
    "lv": ["latvian", "lv_LV"],
    "bn": ["bengali", "bn_IN"],
    "az": ["azerbaijani", "az_AZ"],
    "et": ["estonian", "et_EE"],
    "mk": ["macedonian", "mk_MK"],
    "ne": ["nepali", "ne_NP"],
    "mn": ["mongolian", "mn_MN"],
    "kk": ["kazakh", "kk_KZ"],
    "sw": ["swahili", "sw_KE"],
    "gl": ["galician", "gl_ES"],
    "mr": ["marathi", "mr_IN"],
    "si": ["sinhala", "si_LK"],
    "km": ["khmer", "km_KH"],
    "af": ["afrikaans", "af_ZA"],
    "ka": ["georgian", "ka_GE"],
    "gu": ["gujarati", "gu_IN"],
    "lb": ["luxembourgish", "ps_AF"],
    "tl": ["tagalog", "tl_XX"],
}

def str2bool(string):
    string = string.lower()
    str2val = {"true": True, "false": False}

    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())}, got {string}")

def format_timestamp(seconds: float, always_include_hours: bool = False):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(transcript: Iterator[dict], file: TextIO):
    for i, segment in enumerate(transcript, start=1):
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True)} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True)}\n"
            f"{segment['text'].strip().replace('-->', '->')}\n",
            file=file,
            flush=True,
        )

def filename(path):
    return os.path.splitext(os.path.basename(path))[0]

def load_translator():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    model = MBartForConditionalGeneration.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
    tokenizer = MBart50TokenizerFast.from_pretrained("SnypzZz/Llama2-13b-Language-translate", src_lang="en_XX")
    return model, tokenizer

def get_text_batch(segments:List[dict]):
    text_batch = []
    for i, segment in enumerate(segments):
        text_batch.append(segment['text'])
    return text_batch

def replace_text_batch(segments:List[dict], translated_batch:List[str]):
    for i, segment in enumerate(segments):
        segment['text'] = translated_batch[i]
    return segments

@Gooey(program_name="Subtitle Generator", default_size=(600, 600))
def main():
    parser = GooeyParser(description="Generate subtitles for video files")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default="subtitled", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=True,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=True,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")
    parser.add_argument("--translate_to", type=str, default=None, choices=['ar_AR','en_XX','fr_XX','hi_IN','ja_XX','mr_In'],
    help="Final target language code; Arabic (ar_AR), English (en_XX), Spanish (es_XX), French (fr_XX), Hindi (hi_IN), Japanese (ja_XX), Marathi(mr_IN)")
    
    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    translate_to: str = args.pop("translate_to")
    
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language
    
    model = whisper.load_model(model_name)
    audios = get_audio(args.pop("video"))
    subtitles, detected_language = get_subtitles(
        audios, 
        output_srt or srt_only, 
        output_dir, 
        model,
        args, 
        translate_to=translate_to
    )

    if srt_only:
        return
    
    _translated_to = ""
    if translate_to:
        # for filename
        _translated_to = f"2{translate_to.split('_')[0]}"
        
    for path, srt_path in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}_subtitled_{detected_language}{_translated_to}.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        video = ffmpeg.input(path)
        audio = video.audio

        ffmpeg.concat(
            video.filter('subtitles', srt_path, force_style="FallbackName=NanumGothic,OutlineColour=&H40000000,BorderStyle=3", charenc="UTF-8"), audio, v=1, a=1
        ).output(out_path).run(quiet=True, overwrite_output=True)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")


def get_audio(paths):
    temp_dir = tempfile.gettempdir()

    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths


def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, model:whisper.model.Whisper, args: dict, translate_to: str = None) -> Tuple[dict, str]:
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        print(
            f"Generating subtitles for {filename(path)}... This might take a while."
        )

        warnings.filterwarnings("ignore")
        print("[Step1] detect language (Whisper)")
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels).to(model.device)
        # detect the spoken language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        current_lang = LANG_CODE_MAPPER.get(detected_language, [])
        
        print("[Step2] transcribe (Whisper)")
        if detected_language != "en" and translate_to is not None and translate_to not in current_lang:
            args["task"] = "translate"
            print(f"transcribe-task changed for llama translator")
        result = model.transcribe(audio_path, **args)
        
        if translate_to is not None and translate_to not in current_lang:
            print("[Step3] translate (Llama2)")
            text_batch = get_text_batch(segments=result["segments"])
            translated_batch = translates(translate_to=translate_to, text_batch=text_batch)
            result["segments"] = replace_text_batch(segments=result["segments"], translated_batch=translated_batch)
            print(f"translated to {translate_to}")
        
        with open(srt_path, "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
        subtitles_path[path] = srt_path

    return subtitles_path, detected_language

def translates(translate_to:str, text_batch:List[str]):
    model, tokenizer = load_translator()
    
    model_inputs = tokenizer(text_batch, return_tensors="pt", padding=True)
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[translate_to]
    )
    translated_batch = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    return translated_batch


if __name__ == '__main__':
    main()
