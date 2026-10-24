import json
import random
import subprocess
import sys
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

from model_model import LinearHandshapePredictor
from model_configs import *

# import PyQt5
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.path.dirname(PyQt5.__file__)


FLAG_NONE = 0
FLAG_OK = 1
FLAG_FILLED = 2 # set FLAG_FILLED = 1 if include interpolated data in interpolation next round

class FixedHandshapeDict: 
    def __init__(self) -> None:
        self.charset = [')', ',', '-', '0', '1', '2', '3', '4', '5', '6', '8', ':', ';', '<', '=', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'L', 'M', 'N', 'O', 'P', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '\\', 'b', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '}']
        self.tag2idx_dict = {tag: index for index, tag in enumerate(self.charset)}
        self.idx2tag_dict = {v: k for k, v in self.tag2idx_dict.items()}

    def get_length(self): 
        return len(self.charset)

    def tag2idx(self, tag): 
        return self.tag2idx_dict[tag]
    
    def idx2tag(self, idx): 
        return self.idx2tag_dict[idx]
    
    def batch_map(self, class_index_tensor): 
        class_list = [self.idx2tag(index.item()) for index in class_index_tensor]
        return class_list

def lm_has_side_and_is_at(lm, side):
    # This is a renewed version of lm_has_side_and_is_at, considering the order is not strict
    # Use "Left", and "Right" for side, instead of numbers
    mh = lm.multi_handedness
    if mh is None: 
        return False, 0

    max_score = -1
    max_index = 0
    found = False
    for i, hand in enumerate(mh):
        handedness_dict = MessageToDict(hand)
        classification = handedness_dict["classification"][0]
        if classification["label"] == side:
            if classification["score"] > max_score:
                max_score = classification["score"]
                max_index = i
                found = True
    return found, max_index

def sol2json(d, json_path, side): 
    # with open(json_path, 'w') as fl:
    has, at = lm_has_side_and_is_at(d, side)
    if has: 
        ml = (MessageToDict(d.multi_hand_landmarks[at])["landmark"]) # 0 is one of the hands, do this first
        my_dict = {str(i): (d['x'], d['y'], d['z']) for i, d in enumerate(ml)}
        this_flag = FLAG_OK
    else: 
        my_dict = {str(i): (0, 0, 0) for i in range(21)} # default (0, 0, 0) for all nodes
        this_flag = FLAG_NONE
    # outdict = {"edges": el, "features": my_dict}  # for the current processing, it is not needed to include edges
    outdict = {"features": my_dict, "flag": this_flag}
    # fl.write(json.dumps(outdict, separators=(',', ':')))
    return outdict
    
def dict2array(dict_data):
    feature_list = [dict_data['features'][str(i)] for i in range(21)]
    feature_array = np.array(feature_list)
    return feature_array.reshape(1, 21, 3)


hsdict = FixedHandshapeDict()
def model_predict(model, features, hsdict): 
    this_features = torch.from_numpy(features)
    x = this_features.to(next(model.parameters()).device, dtype=torch.float32)
    with torch.inference_mode():
        hid_rep, pred = model.predict(x, hsdict)
    hid_rep = hid_rep.cpu().numpy()
    return hid_rep, pred


def format_lookup_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value).strip()


def translation_lines(entry):
    translations = entry.get("translation") or {}
    zh_yue = translations.get("ZH-Yue")
    zh_hk = translations.get("ZH-HK") or {}
    if zh_yue is None:
        zh_yue = zh_hk

    def value(language, key, fallback=None):
        if key in language:
            return format_lookup_value(language.get(key))
        return format_lookup_value(fallback.get(key)) if fallback else ""

    cantonese_primary = value(zh_yue, "primary", zh_hk)
    cantonese_secondary = value(zh_yue, "secondary", zh_hk)
    disambiguation = format_lookup_value(zh_hk.get("disambiguation"))
    cantonese = [cantonese_primary]
    if disambiguation:
        cantonese.append(f"({disambiguation})")
    cantonese.append(cantonese_secondary)

    english = translations.get("EN-GB") or {}
    english_primary = format_lookup_value(english.get("primary"))
    english_disambiguation = format_lookup_value(english.get("disambiguation"))
    english_secondary = format_lookup_value(english.get("secondary"))
    english_line = [english_primary]
    if english_disambiguation:
        english_line.append(f"({english_disambiguation})")
    english_line.append(english_secondary)

    entry_id = format_lookup_value(
        entry.get("id")
        or (entry.get("HKU_HKSLLEX_ID") or {}).get("id")
    )
    return [
        " ".join(part for part in cantonese if part),
        " ".join(part for part in english_line if part),
        entry_id,
    ]


class AnimatedWebP:
    MAX_FRAME_SIZE = (240, 135)

    def __init__(self, path):
        self.path = path
        self.frames = []
        self.durations = []
        self.decoded_bytes = 0
        self.decoded_duration = 0
        self.loop_duration = 0
        self.started_at = time.monotonic()
        self.loaded = False
        self.lock = Lock()

    def load(self, on_progress=None):
        with Image.open(self.path) as webp:
            frame_count = max(1, getattr(webp, "n_frames", 1))
            for frame_index in range(frame_count):
                webp.seek(frame_index)
                frame = webp.convert("RGBA")
                frame.thumbnail(self.MAX_FRAME_SIZE, Image.Resampling.LANCZOS)
                with self.lock:
                    self.frames.append(frame.copy())
                    self.durations.append(
                        max(10, int(webp.info.get("duration", 100))) / 1000
                    )
                    self.decoded_bytes += frame.width * frame.height * 4
                    self.decoded_duration += self.durations[-1]
                if on_progress is not None:
                    on_progress((frame_index + 1) / frame_count)

        with self.lock:
            if not self.frames:
                raise ValueError(f"WebP contains no frames: {self.path}")
            self.loop_duration = sum(self.durations)
            self.started_at = time.monotonic()
            self.loaded = True

    def frame_at(self, now):
        with self.lock:
            if not self.frames:
                return None
            position = max(0, now - self.started_at)
            if self.loaded:
                position %= self.loop_duration
            else:
                position %= self.decoded_duration
            for frame, duration in zip(self.frames, self.durations):
                if position < duration:
                    return frame
                position -= duration
            return self.frames[-1]

    @property
    def has_frames(self):
        with self.lock:
            return bool(self.frames)

    @property
    def is_loaded(self):
        with self.lock:
            return self.loaded


class HandshapeVideoLookup:
    LOOKUP_SCHEMA_VERSION = 8
    PRECOMPUTED_RESULT_LIMIT = 12
    ANIMATION_CACHE_LIMIT_BYTES = 512 * 1024 * 1024

    def __init__(self, database_dir, batch_size=6, performance_mode="low-end"):
        self.video_dir = database_dir / "webp"
        self.batch_size = batch_size
        self.performance_mode = performance_mode
        lookup_path = database_dir / "data.lookup.json"
        builder_path = database_dir / "build_handshape_lookup.py"
        source_path = database_dir / "data.json"
        if (
            not lookup_path.is_file()
            or lookup_path.stat().st_mtime_ns < source_path.stat().st_mtime_ns
        ):
            subprocess.run(
                [sys.executable, str(builder_path)],
                check=True,
                cwd=database_dir.parent,
            )
        with lookup_path.open(encoding="utf-8") as data_file:
            lookup_data = json.load(data_file)
        if lookup_data.get("schema_version") != self.LOOKUP_SCHEMA_VERSION:
            subprocess.run(
                [sys.executable, str(builder_path)],
                check=True,
                cwd=database_dir.parent,
            )
            with lookup_path.open(encoding="utf-8") as data_file:
                lookup_data = json.load(data_file)
        if lookup_data.get("schema_version") != self.LOOKUP_SCHEMA_VERSION:
            raise ValueError(f"Unsupported lookup index: {lookup_path}")
        self.entries = lookup_data["entries"]
        self.entry_by_video = {
            entry["video"]: entry for entry in self.entries
        }
        self.precomputed = lookup_data["precomputed"]
        self.matches = []
        self.offset = 0
        self.loaded_until = 0
        self.total_match_count = 0
        self.window_started_at = None
        self.animations = {}
        self.animation_cache = OrderedDict()
        self.animation_cache_bytes = 0
        self.ANIMATION_CACHE_LIMIT_BYTES = (0.5 if self.performance_mode == "low-end" else 1.0) * 1024 * 1024 * 1024
        self.loading_animations = {}
        self.video_load_progress = {}
        self.query_key = None
        self.initial_match_count = 0
        self.continuation_future = None
        self.continuation_query_key = None
        self.continuation_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="handshape-full-lookup"
        )
        self.video_executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="webp-loader"
        )
        self.font_dir = Path(__file__).resolve().parent / "fonts"
        self.font_cache = {}
        self.digit_width_cache = {}

    def get_font(self, filename, size):
        key = (filename, size)
        if key not in self.font_cache:
            self.font_cache[key] = ImageFont.truetype(
                str(self.font_dir / filename), size
            )
        return self.font_cache[key]

    def digit_width(self, font):
        key = (font.getname(), font.size)
        if key not in self.digit_width_cache:
            self.digit_width_cache[key] = max(
                font.getlength(str(digit)) for digit in range(10)
            )
        return self.digit_width_cache[key]

    def update(self, right_codepoint, left_codepoint):
        query_key = self._query_key(right_codepoint, left_codepoint)
        if query_key == self.query_key:
            return

        if self.continuation_future is not None:
            self.continuation_future.cancel()
        self.continuation_future = None
        self.continuation_query_key = None
        self.query_key = query_key
        self._load_query_matches()

    def _load_query_matches(self):
        match_count, video_names = self.precomputed.get(self.query_key, (0, ()))
        self.total_match_count = match_count
        video_names = video_names[:]
        if (
            self.performance_mode == "standard"
            and len(video_names) > self.PRECOMPUTED_RESULT_LIMIT
        ):
            video_names = random.sample(
                video_names, self.PRECOMPUTED_RESULT_LIMIT
            )
        else:
            video_names = video_names[:self.PRECOMPUTED_RESULT_LIMIT]
        self.matches = self._make_matches(video_names)
        self.initial_match_count = len(self.matches)
        self.offset = 0
        self.loaded_until = min(3, len(self.matches))
        self._keep_visible_animations()
        self.window_started_at = None

    def _make_matches(self, video_names):
        self.matches = [
            (
                (0, 0),
                self.entry_by_video[video_name]["id"],
                self.entry_by_video[video_name],
                self.video_dir / video_name,
            )
            for video_name in video_names
        ]
        return self.matches

    def _matching_entries(self, right_codepoint, left_codepoint):
        matches = []
        for entry in self.entries:
            best_rank = None
            for pair_index, (dominant, non_dominant) in enumerate(
                entry["pairs"]
            ):
                rank = None
                if (
                    right_codepoint is not None
                    and left_codepoint is not None
                    and dominant != 0
                    and non_dominant != 0
                    and (
                        (
                            right_codepoint == dominant
                            and left_codepoint == non_dominant
                        )
                        or (
                            left_codepoint == dominant
                            and right_codepoint == non_dominant
                        )
                    )
                ):
                    rank = 0
                elif dominant != 0 and dominant in (
                    right_codepoint,
                    left_codepoint,
                ):
                    rank = 1
                elif non_dominant != 0 and non_dominant in (
                    right_codepoint,
                    left_codepoint,
                ):
                    rank = 2

                if rank is not None:
                    candidate_rank = (pair_index, rank)
                    if best_rank is None or candidate_rank < best_rank:
                        best_rank = candidate_rank
            if best_rank is not None:
                matches.append(
                    (
                        best_rank,
                        entry["id"],
                        entry,
                        self.video_dir / entry["video"],
                    )
                )
        matches.sort(key=lambda match: (match[0], match[3].name))
        return matches

    def _load_remaining_matches(
        self, query_key, selected_video_names, expected_count
    ):
        right, left = query_key.split(",", maxsplit=1)
        right_codepoint = int(right) if right else None
        left_codepoint = int(left) if left else None
        all_matches = self._matching_entries(right_codepoint, left_codepoint)
        if len(all_matches) != expected_count:
            raise ValueError(
                "Precomputed total does not match full handshape lookup."
            )
        return [
            match
            for match in all_matches
            if match[3].name not in selected_video_names
        ]

    def _start_remaining_lookup(self):
        if (
            self.continuation_future is not None
            or self.initial_match_count < self.PRECOMPUTED_RESULT_LIMIT
            or len(self.matches) >= self.total_match_count
        ):
            return
        selected_video_names = {match[3].name for match in self.matches}
        self.continuation_query_key = self.query_key
        self.continuation_future = self.continuation_executor.submit(
            self._load_remaining_matches,
            self.query_key,
            selected_video_names,
            self.total_match_count,
        )

    def _finish_remaining_lookup(self):
        future = self.continuation_future
        if future is None or not future.done():
            return
        query_key = self.continuation_query_key
        self.continuation_future = None
        self.continuation_query_key = None
        remaining = future.result()
        if query_key == self.query_key:
            self.matches.extend(remaining)

    @staticmethod
    def _query_key(right_codepoint, left_codepoint):
        right = "" if right_codepoint is None else str(right_codepoint)
        left = "" if left_codepoint is None else str(left_codepoint)
        return f"{right},{left}"

    def close(self):
        self.continuation_executor.shutdown(wait=False, cancel_futures=True)
        self.video_executor.shutdown(wait=False, cancel_futures=True)

    def set_performance_mode(self, performance_mode, batch_size):
        if performance_mode not in ("low-end", "standard"):
            raise ValueError(f"Unsupported performance mode: {performance_mode}")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive: {batch_size}")
        if (
            performance_mode == self.performance_mode
            and batch_size == self.batch_size
        ):
            return
        self.performance_mode = performance_mode
        self.batch_size = batch_size
        self.ANIMATION_CACHE_LIMIT_BYTES = (0.5 if performance_mode == "low-end" else 1.0) * 1024 * 1024 * 1024
        if self.query_key is not None:
            if self.continuation_future is not None:
                self.continuation_future.cancel()
            self.continuation_future = None
            self.continuation_query_key = None
            self._load_query_matches()
        if self.matches:
            self.loaded_until = min(
                len(self.matches),
                max(
                    self.offset + 3,
                    (self.offset // batch_size + 1) * batch_size,
                ),
            )

    def _cache_animation(self, entry_id, animation, protected_ids):
        previous = self.animation_cache.pop(entry_id, None)
        if previous is not None:
            self.animation_cache_bytes -= previous.decoded_bytes
        if animation.decoded_bytes > self.ANIMATION_CACHE_LIMIT_BYTES:
            return

        self.animation_cache[entry_id] = animation
        self.animation_cache_bytes += animation.decoded_bytes
        while self.animation_cache_bytes > self.ANIMATION_CACHE_LIMIT_BYTES:
            evict_id = next(
                (
                    cached_id
                    for cached_id in self.animation_cache
                    if cached_id not in protected_ids
                ),
                None,
            )
            if evict_id is None:
                break
            evicted = self.animation_cache.pop(evict_id)
            self.animation_cache_bytes -= evicted.decoded_bytes

    def _keep_visible_animations(self):
        visible = self.matches[self.offset:min(self.offset + 3, self.loaded_until)]
        active_ids = {match[1] for match in visible}
        for entry_id, (future, animation) in list(
            self.loading_animations.items()
        ):
            if future.done():
                future.result()
                del self.loading_animations[entry_id]
                self.video_load_progress.pop(entry_id, None)
                if self.animation_cache_bytes + animation.decoded_bytes <= self.ANIMATION_CACHE_LIMIT_BYTES:
                    self._cache_animation(entry_id, animation, active_ids)
                else:
                    # Stop caching when limit reached
                    pass
            elif entry_id not in active_ids and future.cancel():
                del self.loading_animations[entry_id]
                self.video_load_progress.pop(entry_id, None)

        self.animations = {}
        for _, entry_id, _, video_path in visible:
            animation = self.animation_cache.get(entry_id)
            if animation is not None:
                self.animation_cache.move_to_end(entry_id)
                self.animations[entry_id] = animation
                continue

            loading = self.loading_animations.get(entry_id)
            if loading is None:
                # Only start loading if there is space for it
                # Note: we don't know the size yet, but we can check if we are already at the limit
                # Or just load it and let _cache_animation handle it.
                # The requirement says "stop caching more videos when loading" when reached.
                # So we load it to show it, but don't put it in the LRU cache.
                animation = AnimatedWebP(video_path)
                progress_state = [0.0]
                self.video_load_progress[entry_id] = progress_state
                future = self.video_executor.submit(
                    animation.load,
                    lambda progress, state=progress_state: state.__setitem__(
                        0, progress
                    ),
                )
                self.loading_animations[entry_id] = (future, animation)
            else:
                _, animation = loading
            self.animations[entry_id] = animation

    def visible_matches(self, now):
        self._finish_remaining_lookup()
        if not self.matches:
            return []

        if (
            self.loaded_until < len(self.matches)
            and self.offset + 3 >= self.loaded_until
        ):
            self.loaded_until = min(
                self.loaded_until + self.batch_size, len(self.matches)
            )
        self._keep_visible_animations()

        lead = self.matches[self.offset]
        lead_animation = self.animations.get(lead[1])
        if (
            not isinstance(lead_animation, AnimatedWebP)
            or not lead_animation.has_frames
        ):
            return self.matches[
                self.offset:min(self.offset + 3, self.loaded_until)
            ]
        if self.window_started_at is None and lead_animation.is_loaded:
            self.window_started_at = now
            lead_animation.started_at = now
        if self.offset >= self.initial_match_count - 2:
            self._start_remaining_lookup()
        if (
            lead_animation.is_loaded
            and self.offset < self.total_match_count - 1
            and self.offset < len(self.matches) - 1
            and self.window_started_at is not None
            and now - self.window_started_at >= lead_animation.loop_duration
        ):
            self.offset += 1
            self._keep_visible_animations()
            self.window_started_at = None
        return self.matches[self.offset:min(self.offset + 3, self.loaded_until)]

    def _wrap_text(self, text, draw, max_width, latin_font, cjk_font):
        lines = []
        current = ""
        for character in text:
            candidate = current + character
            if current and self._text_width(
                candidate, draw, latin_font, cjk_font
            ) > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    @staticmethod
    def _is_cjk(character):
        codepoint = ord(character)
        return (
            0x2E80 <= codepoint <= 0xA4CF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0xFE30 <= codepoint <= 0xFE4F
            or 0xFF00 <= codepoint <= 0xFFEF
            or codepoint >= 0x20000
        )

    def _text_width(self, text, draw, latin_font, cjk_font):
        digit_width = self.digit_width(latin_font)
        return sum(
            digit_width
            if character.isdigit()
            else draw.textlength(
                character,
                font=cjk_font if self._is_cjk(character) else latin_font,
            )
            for character in text
        )

    def _draw_text(self, draw, text, position, latin_font, cjk_font, fill):
        digit_width = self.digit_width(latin_font)
        text_x, text_y = position
        for character in text:
            is_digit = character.isdigit()
            font = cjk_font if self._is_cjk(character) else latin_font
            draw.text(
                (
                    text_x + (digit_width - latin_font.getlength(character)) / 2
                    if is_digit
                    else text_x,
                    text_y,
                ),
                character,
                font=font,
                fill=fill,
            )
            text_x += (
                digit_width
                if is_digit
                else draw.textlength(character, font=font)
            )
        return text_x - position[0]

    @staticmethod
    def indicator_layout(screen_width):
        panel_width = max(1, round(screen_width * 0.15))
        margin = max(4, panel_width // 24)
        gap = max(4, margin // 2)
        size = max(18, (panel_width - 2 * margin - gap - 20) // 2)
        return panel_width, margin, gap, size, size + 2 * margin

    def render_sidebar(self, frame, screen_width, screen_height):
        now = time.monotonic()
        visible = self.visible_matches(now)
        panel_width, margin, _, _, header_height = self.indicator_layout(
            screen_width
        )
        footer_height = max(24, screen_height // 30)
        base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, panel_width, screen_height), fill=(8, 8, 8, 225))
        base = Image.alpha_composite(base, overlay)
        draw = ImageDraw.Draw(base)
        font_size = max(11, screen_width // 150)
        latin_font = self.get_font("PublicSans-Regular.ttf", font_size)
        cjk_font = self.get_font("GenKiGothic2TW-R.otf", font_size)
        inner_width = max(1, panel_width - 2 * margin)
        content_top = header_height
        content_height = max(
            1, screen_height - content_top - footer_height
        )
        card_height = max(1, (content_height - 4 * margin) // 3)
        caption_font_size = max(font_size + 4, screen_width // 100)
        caption_latin_font = self.get_font(
            "PublicSans-Regular.ttf", caption_font_size
        )
        caption_cjk_font = self.get_font(
            "GenKiGothic2TW-R.otf", caption_font_size
        )
        cantonese_font_size = max(caption_font_size + 5, screen_width // 85)
        cantonese_latin_font = self.get_font(
            "PublicSans-Regular.ttf", cantonese_font_size
        )
        cantonese_cjk_font = self.get_font(
            "GenKiGothic2TW-R.otf", cantonese_font_size
        )
        id_font_size = max(10, font_size - 2)
        id_latin_font = self.get_font("PublicSans-Regular.ttf", id_font_size)
        id_cjk_font = self.get_font("GenKiGothic2TW-R.otf", id_font_size)
        draw = ImageDraw.Draw(base)
        shown_matches = len(visible)
        remaining_matches = max(
            0, len(self.matches) - self.offset - shown_matches
        )
        footer_text = (
            f"{self.total_match_count} matches  |  {remaining_matches} remaining"
        )
        footer_width = self._text_width(
            footer_text, draw, latin_font, cjk_font
        )
        self._draw_text(
            draw,
            footer_text,
            (max(0, (panel_width - footer_width) / 2), screen_height - footer_height),
            latin_font,
            cjk_font,
            "white",
        )

        if not visible:
            instruction = "Hold to show matches"
            instruction_lines = self._wrap_text(
                instruction,
                draw,
                inner_width,
                caption_latin_font,
                caption_cjk_font,
            )
            instruction_y = content_top + margin
            instruction_line_height = (
                caption_font_size + max(2, caption_font_size // 4)
            )
            for line in instruction_lines:
                self._draw_text(
                    draw,
                    line,
                    (margin, instruction_y),
                    caption_latin_font,
                    caption_cjk_font,
                    "white",
                )
                instruction_y += instruction_line_height
        else:
            for index, (_, entry_id, entry, _) in enumerate(visible):
                card_top = content_top + margin + index * (card_height + margin)
                video_box_height = int(card_height * 0.54)
                animation = self.animations.get(entry_id)
                if animation is not None:
                    source_frame = animation.frame_at(now)
                else:
                    source_frame = None
                if source_frame is not None:
                    scale = min(
                        panel_width / source_frame.width,
                        video_box_height / source_frame.height,
                    )
                    video_size = (
                        max(1, round(source_frame.width * scale)),
                        max(1, round(source_frame.height * scale)),
                    )
                    video = source_frame.resize(
                        video_size, Image.Resampling.LANCZOS
                    )
                    base.alpha_composite(video, (0, card_top))
                else:
                    progress_state = self.video_load_progress.get(entry_id)
                    progress = (
                        progress_state[0] if progress_state is not None else 0.0
                    )
                    placeholder = f"Loading {round(progress * 100)}%"
                    placeholder_box = draw.textbbox(
                        (0, 0), placeholder, font=latin_font
                    )
                    placeholder_width = (
                        placeholder_box[2] - placeholder_box[0]
                    )
                    placeholder_height = (
                        placeholder_box[3] - placeholder_box[1]
                    )
                    draw.rectangle(
                        (
                            0,
                            card_top,
                            panel_width,
                            card_top + video_box_height,
                        ),
                        fill=(20, 20, 20, 255),
                    )
                    draw.text(
                        (
                            (panel_width - placeholder_width) // 2
                            - placeholder_box[0],
                            card_top
                            + (video_box_height - placeholder_height) // 2
                            - placeholder_box[1],
                        ),
                        placeholder,
                        font=latin_font,
                        fill=(255, 255, 255, 255),
                    )
                    progress_margin = max(8, margin * 2)
                    progress_left = progress_margin
                    progress_right = panel_width - progress_margin
                    progress_y = card_top + video_box_height - max(
                        8, margin
                    )
                    draw.rectangle(
                        (
                            progress_left,
                            progress_y,
                            progress_right,
                            progress_y + 4,
                        ),
                        fill=(90, 90, 90, 255),
                    )
                    draw.rectangle(
                        (
                            progress_left,
                            progress_y,
                            progress_left
                            + round(
                                (progress_right - progress_left) * progress
                            ),
                            progress_y + 4,
                        ),
                        fill=(80, 210, 120, 255),
                    )

                text_top = card_top + video_box_height + margin
                lines = []
                translations = translation_lines(entry)
                for translation_index, text in enumerate(translations):
                    if text:
                        if translation_index == 0:
                            text_latin_font = cantonese_latin_font
                            text_cjk_font = cantonese_cjk_font
                        elif translation_index == 2:
                            text_latin_font = id_latin_font
                            text_cjk_font = id_cjk_font
                        else:
                            text_latin_font = caption_latin_font
                            text_cjk_font = caption_cjk_font
                        lines.extend(
                            (
                                line,
                                text_latin_font,
                                text_cjk_font,
                                text_latin_font.size
                                + max(2, text_latin_font.size // 4),
                            )
                            for line in self._wrap_text(
                                text, draw, inner_width,
                                text_latin_font, text_cjk_font,
                            )
                        )
                for line, line_latin_font, line_cjk_font, line_height in lines:
                    if text_top + line_height > min(
                        screen_height, card_top + card_height
                    ):
                        break
                    self._draw_text(
                        draw,
                        line,
                        (margin, text_top),
                        line_latin_font,
                        line_cjk_font,
                        (255, 255, 255, 255),
                    )
                    text_top += line_height

        rgb_frame = np.asarray(base.convert("RGB"))
        return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)


class ListBuffer: 
    def __init__(self, buffer_size) -> None:
        self.buffer = deque(maxlen=buffer_size)
    def append(self, item): 
        self.buffer.append(item)
    def get(self): 
        return list(self.buffer)
    def stack_and_get(self): 
        return np.stack(self.buffer, axis=0)
class HandDetection:
    def __init__(
        self,
        model,
        video_path=None,
        lookup_batch_size=6,
        performance_mode="low-end",
    ):
        self.model = model
        self.performance_mode = performance_mode
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.handshape_dir = Path(__file__).resolve().parent / "hs_char"
        self.lookup = HandshapeVideoLookup(
            Path(__file__).resolve().parent / "database",
            batch_size=lookup_batch_size,
            performance_mode=performance_mode,
        )
        self.pending_handshapes = None
        self.handshape_images = {}
        self.missing_handshape_images = set()
        self.saved_frames_hidrep_right = ListBuffer(10)
        self.saved_frames_hidrep_left = ListBuffer(10)
        self.hidden_plot_right = None
        self.hidden_plot_left = None
        self.fallback_reason = None
        try_idxs = [-1, 0, 1, 2, -2]

        for idx in try_idxs:
            print(f"Trying to open webcam {idx}...")
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                break
            self.cap.release()
        if not self.cap.isOpened():
            self.fallback_reason = (
                "No webcam detected. Playing fallback video: "
                f"{Path(video_path).name if video_path else 'unavailable'}"
            )
            print(self.fallback_reason)
            if video_path and Path(video_path).is_file():
                self.cap = cv2.VideoCapture(str(video_path))
            else:
                raise FileNotFoundError(
                    f"No webcam found and video file is unavailable: {video_path}"
                )
            if not self.cap.isOpened():
                self.cap.release()
                raise RuntimeError(f"Unable to open video file: {video_path}")
        # Request 1080p capture for a sharper fullscreen render when supported.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Store the actual frame width and height
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def overlay_handshape(self, frame, prediction, position=(50, 50), size=(120, 120)):
        """Overlay predicted handshape image on the frame with a white background."""
        handshape_path = self.handshape_dir / f"{ord(prediction):03d}.png"
        cache_key = (prediction, size)
        if cache_key not in self.handshape_images and cache_key not in self.missing_handshape_images:
            if not handshape_path.is_file():
                print(f"Handshape image not found: {handshape_path}")
                self.missing_handshape_images.add(cache_key)
            else:
                handshape_img = cv2.imread(str(handshape_path), cv2.IMREAD_UNCHANGED)
                if handshape_img is None:
                    raise ValueError(f"Unable to read handshape image: {handshape_path}")
                handshape_img = cv2.resize(handshape_img, size)
                if handshape_img.ndim == 2:
                    handshape_img = cv2.cvtColor(handshape_img, cv2.COLOR_GRAY2BGR)
                elif handshape_img.shape[2] == 4:
                    handshape_img = handshape_img[:, :, :3]

                white_bg = np.full((size[1] + 10, size[0] + 10, 3), 255, dtype=np.uint8)
                white_bg[5:5 + size[1], 5:5 + size[0]] = handshape_img
                self.handshape_images[cache_key] = white_bg

        handshape_tile = self.handshape_images.get(cache_key)
        if handshape_tile is not None:
            x, y = position
            h, w = handshape_tile.shape[:2]
            frame[y:y + h, x:x + w] = handshape_tile

        return frame

    def plot_hidden_representation(self, hidden_rep, size=(250, 150)):
        """Render hidden activations without creating per-frame Matplotlib figures."""
        canvas = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        num_frames, num_dims = hidden_rep.shape
        if num_frames == 0 or num_dims == 0:
            return canvas

        values = np.asarray(hidden_rep, dtype=np.float32)
        value_min = float(values.min())
        value_max = float(values.max())
        value_range = value_max - value_min
        if value_range == 0:
            value_range = 1.0

        x_coords = np.linspace(0, size[0] - 1, num_frames).astype(np.int32)
        colors = cv2.applyColorMap(
            np.linspace(0, 255, num_dims, dtype=np.uint8).reshape(1, -1),
            cv2.COLORMAP_VIRIDIS,
        )[0]
        for dim, color in enumerate(colors):
            y_coords = size[1] - 1 - (
                (values[:, dim] - value_min) / value_range * (size[1] - 1)
            ).astype(np.int32)
            points = np.column_stack((x_coords, y_coords)).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], False, tuple(int(c) for c in color), 2)

        return canvas

    def process_detection_frame(self, hands, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.flip(rgb_frame, 1)
        results = hands.process(rgb_frame)

        right_lm = sol2json(results, None, "Right")
        left_lm = sol2json(results, None, "Left")
        hid_rep_right, pred_right = model_predict(
            self.model, dict2array(right_lm), hsdict
        )
        hid_rep_left, pred_left = model_predict(
            self.model, dict2array(left_lm), hsdict
        )

        annotated_frame = cv2.flip(frame.copy(), 1)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        return (
            annotated_frame,
            hid_rep_right[0],
            hid_rep_left[0],
            pred_right[0],
            pred_left[0],
            right_lm["flag"],
            left_lm["flag"],
        )

    def read_and_process_detection_frame(self, hands):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return self.process_detection_frame(hands, frame)

    def update_handshape_matches(self, observed_handshapes):
        if observed_handshapes == (None, None):
            if self.pending_handshapes is not None:
                self.pending_handshapes = None
                self.lookup.update(None, None)
        elif observed_handshapes != self.pending_handshapes:
            self.pending_handshapes = observed_handshapes
            self.lookup.update(*observed_handshapes)

    def detect(self):
        detection_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hand-detection"
        )
        try:
            window_name = "Hand Detection with Overlays"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            cv2.waitKey(1)
            _, _, screen_width, screen_height = cv2.getWindowImageRect(window_name)
            if screen_width <= 0 or screen_height <= 0:
                raise RuntimeError(
                    f"Unable to determine fullscreen render size: "
                    f"{screen_width}x{screen_height}"
                )
            resolution_percent = 100
            with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
                detection_future = None
                latest_frame = None
                hidden_plot_size = (350, 150)
                pred_right = pred_left = chr(0)
                right_flag = left_flag = FLAG_NONE
                observed_handshapes = (None, None)
                while True:
                    if detection_future is not None and detection_future.done():
                        detection_result = detection_future.result()
                        detection_future = None
                        if detection_result is None:
                            print("End of video or camera disconnected.")
                            break
                        (
                            latest_frame,
                            hid_rep_right,
                            hid_rep_left,
                            pred_right,
                            pred_left,
                            right_flag,
                            left_flag,
                        ) = detection_result
                        self.saved_frames_hidrep_right.append(hid_rep_right)
                        self.saved_frames_hidrep_left.append(hid_rep_left)
                        self.hidden_plot_right = self.plot_hidden_representation(
                            self.saved_frames_hidrep_right.stack_and_get(),
                            size=hidden_plot_size,
                        )
                        self.hidden_plot_left = self.plot_hidden_representation(
                            self.saved_frames_hidrep_left.stack_and_get(),
                            size=hidden_plot_size,
                        )
                        observed_handshapes = (
                            ord(pred_right) if right_flag == FLAG_OK else None,
                            ord(pred_left) if left_flag == FLAG_OK else None,
                        )

                    if detection_future is None:
                        detection_future = detection_executor.submit(
                            self.read_and_process_detection_frame, hands
                        )

                    self.update_handshape_matches(observed_handshapes)
                    if latest_frame is None:
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), ord("Q")):
                            break
                        continue

                    annotated_frame = latest_frame.copy()
                    height, width = annotated_frame.shape[:2]
                    banner_height = max(56, height // 12)

                    footer_height = max(34, height // 20)
                    plot_margin = max(20, width // 40)
                    plot_x = max(0, width - hidden_plot_size[0] - plot_margin)
                    plot_y_right = banner_height + plot_margin
                    plot_y_left = plot_y_right + hidden_plot_size[1] + plot_margin
                    if (
                        self.hidden_plot_right is not None
                        and self.hidden_plot_left is not None
                        and plot_y_left + hidden_plot_size[1]
                        <= height - footer_height - plot_margin
                    ):
                        annotated_frame[
                            plot_y_right:plot_y_right + hidden_plot_size[1],
                            plot_x:plot_x + hidden_plot_size[0],
                        ] = self.hidden_plot_right
                        annotated_frame[
                            plot_y_left:plot_y_left + hidden_plot_size[1],
                            plot_x:plot_x + hidden_plot_size[0],
                        ] = self.hidden_plot_left

                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
                    cv2.rectangle(
                        overlay,
                        (0, height - footer_height),
                        (width, height),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.addWeighted(overlay, 0.65, annotated_frame, 0.35, 0, annotated_frame)

                    title = "HKU Linguistics: HKSL Handshape Detection Model Prototype Demo"
                    footer = (
                        f"[W] +10%  [E] -10%  Resolution: {resolution_percent}%  "
                        f"[1] Low-end{'*' if self.performance_mode == 'low-end' else ''}  "
                        f"[2] Standard{'*' if self.performance_mode == 'standard' else ''}  "
                        f"[S] Cap+ [D] Cap- [R] Purge  "
                        f"[Q] Quit"
                    )
                    text_image = Image.fromarray(
                        cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    )
                    text_draw = ImageDraw.Draw(text_image)
                    title_font_size = min(42, max(18, banner_height // 2))
                    title_font = self.lookup.get_font(
                        "PublicSans-Regular.ttf", title_font_size
                    )
                    title_cjk_font = self.lookup.get_font(
                        "GenKiGothic2TW-R.otf", title_font_size
                    )
                    fallback_font_size = max(12, title_font_size // 2)
                    fallback_font = self.lookup.get_font(
                        "PublicSans-Regular.ttf", fallback_font_size
                    )
                    fallback_cjk_font = self.lookup.get_font(
                        "GenKiGothic2TW-R.otf", fallback_font_size
                    )
                    footer_font_size = max(14, footer_height // 3)
                    footer_font = self.lookup.get_font(
                        "PublicSans-Regular.ttf", footer_font_size
                    )
                    footer_cjk_font = self.lookup.get_font(
                        "GenKiGothic2TW-R.otf", footer_font_size
                    )
                    title_width = self.lookup._text_width(
                        title, text_draw, title_font, title_cjk_font
                    )
                    footer_width = self.lookup._text_width(
                        footer, text_draw, footer_font, footer_cjk_font
                    )
                    self.lookup._draw_text(
                        text_draw,
                        title,
                        (
                            max(10, width - title_width - 10),
                            (
                                max(
                                    4,
                                    (
                                        banner_height
                                        - title_font_size
                                        - fallback_font_size
                                        - 12
                                    )
                                    // 2,
                                )
                                if self.fallback_reason
                                else (banner_height - title_font_size) // 2
                            ),
                        ),
                        title_font,
                        title_cjk_font,
                        (255, 255, 255),
                    )
                    if self.fallback_reason:
                        fallback_width = self.lookup._text_width(
                            self.fallback_reason,
                            text_draw,
                            fallback_font,
                            fallback_cjk_font,
                        )
                        self.lookup._draw_text(
                            text_draw,
                            self.fallback_reason,
                            (
                                max(10, (width - fallback_width) // 2),
                                banner_height - fallback_font_size - 4,
                            ),
                            fallback_font,
                            fallback_cjk_font,
                            (255, 220, 120),
                        )
                    self.lookup._draw_text(
                        text_draw,
                        footer,
                        (max(10, (width - footer_width) // 2), height - footer_height + (footer_height - footer_font_size) // 2),
                        footer_font,
                        footer_cjk_font,
                        (255, 255, 255),
                    )
                    mem_mib = self.lookup.animation_cache_bytes / 1024 / 1024
                    mem_limit_mib = self.lookup.ANIMATION_CACHE_LIMIT_BYTES / 1024 / 1024
                    mem_text = f"MEM: {mem_mib:.1f}/{mem_limit_mib:.1f} MiB"
                    mem_width = self.lookup._text_width(
                        mem_text, text_draw, footer_font, footer_cjk_font
                    )
                    self.lookup._draw_text(
                        text_draw,
                        mem_text,
                        (width - mem_width - 10, height - footer_height + (footer_height - footer_font_size) // 2),
                        footer_font,
                        footer_cjk_font,
                        (255, 255, 255),
                    )
                    annotated_frame = cv2.cvtColor(
                        np.asarray(text_image), cv2.COLOR_RGB2BGR
                    )

                    render_width = max(1, screen_width * resolution_percent // 100)
                    render_height = max(1, screen_height * resolution_percent // 100)
                    render_frame = cv2.resize(
                        annotated_frame,
                        (render_width, render_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    if resolution_percent < 100:
                        render_frame = cv2.resize(
                            render_frame,
                            (screen_width, screen_height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    render_frame = self.lookup.render_sidebar(
                        render_frame, screen_width, screen_height
                    )
                    _, indicator_margin, indicator_gap, indicator_size, header_height = (
                        self.lookup.indicator_layout(screen_width)
                    )
                    indicator_y = max(4, (header_height - indicator_size) // 2)
                    render_frame = self.overlay_handshape(
                        render_frame,
                        pred_left if left_flag == FLAG_OK else chr(0),
                        position=(indicator_margin, indicator_y),
                        size=(indicator_size, indicator_size),
                    )
                    render_frame = self.overlay_handshape(
                        render_frame,
                        pred_right if right_flag == FLAG_OK else chr(0),
                        position=(
                            indicator_margin + indicator_size + 10 + indicator_gap,
                            indicator_y,
                        ),
                        size=(indicator_size, indicator_size),
                    )
                    cv2.imshow(window_name, render_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')):
                        break
                    if key in (ord('w'), ord('W')):
                        resolution_percent = min(100, resolution_percent + 10)
                    elif key in (ord('e'), ord('E')):
                        resolution_percent = max(10, resolution_percent - 10)
                    elif key == ord('1'):
                        self.performance_mode = "low-end"
                        self.lookup.set_performance_mode("low-end", 6)
                    elif key == ord('2'):
                        self.performance_mode = "standard"
                        self.lookup.set_performance_mode("standard", 12)
                    elif key == ord('s') or key == ord('S'):
                        self.lookup.ANIMATION_CACHE_LIMIT_BYTES += 100 * 1024 * 1024
                    elif key == ord('d') or key == ord('D'):
                        self.lookup.ANIMATION_CACHE_LIMIT_BYTES = max(
                            0, self.lookup.ANIMATION_CACHE_LIMIT_BYTES - 100 * 1024 * 1024
                        )
                    elif key == ord('r') or key == ord('R'):
                        self.lookup.animation_cache.clear()
                        self.lookup.animation_cache_bytes = 0
        finally:
            detection_executor.shutdown(wait=True)
            self.cap.release()
            self.lookup.close()
            cv2.destroyAllWindows()

def main():
    import argparse
    import traceback

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--performance-mode",
        choices=("low-end", "standard"),
        default="low-end",
    )
    args = parser.parse_args()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LinearHandshapePredictor(
            input_dim=in_dim,
            enc_lat_dims=enc_lat_dims,
            hid_dim=hid_dim,
            dec_lat_dims=dec_lat_dims,
            output_dim=out_dim
        )
        model_path = Path(__file__).resolve().parent / "597.pt"
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        video_path = (
            Path(__file__).resolve().parent
            / "B_01_079-NOTHING-0KC7-67.mp4"
        )
        lookup_batch_size = 6 if args.performance_mode == "low-end" else 12
        detector = HandDetection(
            model=model,
            video_path=video_path,
            lookup_batch_size=lookup_batch_size,
            performance_mode=args.performance_mode,
        )
        detector.detect()
    except Exception:
        traceback.print_exc()
        try:
            input("Demo failed. Press Enter to exit...")
        except EOFError:
            pass
        raise SystemExit(1) from None

if __name__ == '__main__':
    main()
