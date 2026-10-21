import json
import time
from collections import deque
from pathlib import Path

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


def parse_codepoint_pairs(notation):
    pairs = []
    for group in notation.replace(".", ",").split(","):
        values = [int(value) for value in group.split(";") if value]
        for index in range(0, len(values), 2):
            dominant = values[index]
            non_dominant = values[index + 1] if index + 1 < len(values) else 0
            pairs.append((dominant, non_dominant))
    return pairs


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
        (entry.get("HKU_HKSLLEX_ID") or {}).get("id")
    )
    return [
        " ".join(part for part in cantonese if part),
        " ".join(part for part in english_line if part),
        entry_id,
    ]


class AnimatedWebP:
    def __init__(self, path):
        self.path = path
        self.frames = []
        self.durations = []
        with Image.open(path) as webp:
            for frame_index in range(getattr(webp, "n_frames", 1)):
                webp.seek(frame_index)
                self.frames.append(webp.convert("RGBA").copy())
                duration = max(10, int(webp.info.get("duration", 100))) / 1000
                self.durations.append(duration)

        if not self.frames:
            raise ValueError(f"WebP contains no frames: {path}")

        self.loop_duration = sum(self.durations)
        self.started_at = time.monotonic()

    def frame_at(self, now):
        position = (now - self.started_at) % self.loop_duration
        for frame, duration in zip(self.frames, self.durations):
            if position < duration:
                return frame
            position -= duration
        return self.frames[-1]


class HandshapeVideoLookup:
    def __init__(self, database_dir):
        self.video_dir = database_dir / "webp"
        with (database_dir / "data.json").open(encoding="utf-8") as data_file:
            self.entries = json.load(data_file)
        self.matches = []
        self.offset = 0
        self.window_started_at = None
        self.animations = {}
        self.font_path = self._find_cjk_font()

    @staticmethod
    def _find_cjk_font():
        windows_fonts = Path("C:/Windows/Fonts")
        font_candidates = (
            windows_fonts / "msjh.ttc",
            windows_fonts / "mingliu.ttc",
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        )
        for font_path in font_candidates:
            if font_path.is_file():
                return font_path
        raise FileNotFoundError(
            "No CJK font found."
        )

    def _matching_entries(self, right_codepoint, left_codepoint):
        matches = []
        for entry in self.entries:
            record = entry.get("HKU_HKSLLEX_ID") or {}
            if record.get("status") == "under_review":
                continue

            notation = (entry.get("cuhkcslds_HandshapeFont") or {}).get(
                "CodepointNotation"
            )
            if not notation:
                continue

            best_rank = None
            for pair_index, (dominant, non_dominant) in enumerate(
                parse_codepoint_pairs(notation)
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
                elif right_codepoint in (dominant, non_dominant):
                    rank = 2

                if rank is not None:
                    candidate_rank = (pair_index, rank)
                    if best_rank is None or candidate_rank < best_rank:
                        best_rank = candidate_rank

            if best_rank is None:
                continue

            video_name = (entry.get("videoFile") or {}).get("nameGIF")
            if not video_name:
                continue
            video_path = self.video_dir / Path(video_name).name
            if not video_path.is_file():
                print(f"Matching WebP not found: {video_path}")
                continue
            matches.append((best_rank, record.get("id", ""), entry, video_path))

        matches.sort(key=lambda match: (match[0], match[1]))
        return matches

    def update(self, right_codepoint, left_codepoint):
        matches = self._matching_entries(right_codepoint, left_codepoint)
        if [match[1] for match in matches] == [match[1] for match in self.matches]:
            return

        self.matches = matches
        self.offset = 0
        self._keep_visible_animations()
        self.window_started_at = time.monotonic() if matches else None
        if matches:
            self.animations[matches[0][1]].started_at = self.window_started_at

    def _keep_visible_animations(self):
        visible = self.matches[self.offset:self.offset + 3]
        active_ids = {match[1] for match in visible}
        self.animations = {
            entry_id: animation
            for entry_id, animation in self.animations.items()
            if entry_id in active_ids
        }
        for _, entry_id, _, video_path in visible:
            if entry_id not in self.animations:
                self.animations[entry_id] = AnimatedWebP(video_path)

    def visible_matches(self, now):
        if not self.matches:
            return []

        lead = self.matches[self.offset]
        lead_animation = self.animations[lead[1]]
        if (
            self.offset < len(self.matches) - 1
            and self.window_started_at is not None
            and now - self.window_started_at >= lead_animation.loop_duration
        ):
            self.offset += 1
            self._keep_visible_animations()
            self.window_started_at = time.monotonic()
            self.animations[self.matches[self.offset][1]].started_at = (
                self.window_started_at
            )
        return self.matches[self.offset:self.offset + 3]

    @staticmethod
    def _wrap_text(text, font, draw, max_width):
        lines = []
        current = ""
        for character in text:
            candidate = current + character
            if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                lines.append(current)
                current = character
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def render_sidebar(self, frame, screen_width, screen_height):
        now = time.monotonic()
        visible = self.visible_matches(now)
        panel_width = max(1, round(screen_width * 0.15))
        margin = max(4, panel_width // 24)
        card_height = max(1, (screen_height - 4 * margin) // 3)
        base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle((0, 0, panel_width, screen_height), fill=(8, 8, 8, 225))
        base = Image.alpha_composite(base, overlay)
        draw = ImageDraw.Draw(base)
        font_size = max(11, screen_width // 150)
        font = ImageFont.truetype(str(self.font_path), font_size)
        line_height = font_size + max(2, font_size // 4)
        inner_width = max(1, panel_width - 2 * margin)

        if not visible:
            draw.text((margin, margin), "No matching signs", font=font, fill="white")
        else:
            for index, (_, entry_id, entry, _) in enumerate(visible):
                card_top = margin + index * (card_height + margin)
                video_box_height = int(card_height * 0.58)
                animation = self.animations[entry_id]
                source_frame = animation.frame_at(now)
                scale = min(
                    panel_width / source_frame.width,
                    video_box_height / source_frame.height,
                )
                video_size = (
                    max(1, round(source_frame.width * scale)),
                    max(1, round(source_frame.height * scale)),
                )
                video = source_frame.resize(video_size, Image.Resampling.LANCZOS)
                base.alpha_composite(video, (0, card_top))

                text_top = card_top + video_box_height + margin
                lines = []
                for text in translation_lines(entry):
                    if text:
                        lines.extend(
                            self._wrap_text(text, font, draw, inner_width)
                        )
                for line in lines:
                    if text_top + line_height > min(
                        screen_height, card_top + card_height
                    ):
                        break
                    draw.text(
                        (margin, text_top),
                        line,
                        font=font,
                        fill=(255, 255, 255, 255),
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
    def __init__(self, model, video_path=None):
        self.model = model
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.handshape_dir = Path(__file__).resolve().parent / "hs_char"
        self.lookup = HandshapeVideoLookup(
            Path(__file__).resolve().parent / "database"
        )
        self.pending_handshapes = None
        self.pending_handshapes_since = None
        self.active_handshapes = None
        self.handshape_images = {}
        self.missing_handshape_images = set()
        self.saved_frames_hidrep_right = ListBuffer(10)
        self.saved_frames_hidrep_left = ListBuffer(10)
        try_idxs = [-1, 0, 1, 2, -2]

        for idx in try_idxs:
            print(f"Trying to open webcam {idx}...")
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                break
            self.cap.release()
        if not self.cap.isOpened():
            print("No webcam found! Loading a video file...")
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

    def detect(self):
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
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("End of video or camera disconnected.")
                        break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = cv2.flip(rgb_frame, 1)  # Mirror effect
                    results = hands.process(rgb_frame)

                    right_lm = sol2json(results, None, "Right")
                    left_lm = sol2json(results, None, "Left")
                    right_lm_array = dict2array(right_lm)
                    left_lm_array = dict2array(left_lm)

                    hid_rep_right, pred_right = model_predict(self.model, right_lm_array, hsdict)
                    hid_rep_left, pred_left = model_predict(self.model, left_lm_array, hsdict)
                    self.saved_frames_hidrep_right.append(hid_rep_right[0])
                    self.saved_frames_hidrep_left.append(hid_rep_left[0])

                    observed_handshapes = (
                        ord(pred_right[0]) if right_lm["flag"] == FLAG_OK else None,
                        ord(pred_left[0]) if left_lm["flag"] == FLAG_OK else None,
                    )
                    if observed_handshapes == (None, None):
                        self.pending_handshapes = None
                        self.pending_handshapes_since = None
                    else:
                        now = time.monotonic()
                        if observed_handshapes != self.pending_handshapes:
                            self.pending_handshapes = observed_handshapes
                            self.pending_handshapes_since = now
                        elif (
                            self.pending_handshapes_since is not None
                            and now - self.pending_handshapes_since >= 1
                            and observed_handshapes != self.active_handshapes
                        ):
                            self.active_handshapes = observed_handshapes
                            self.lookup.update(*observed_handshapes)

                    annotated_frame = cv2.flip(frame.copy(), 1)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                annotated_frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                self.mp_drawing_styles.get_default_hand_connections_style()
                            )

                    height, width = annotated_frame.shape[:2]
                    banner_height = max(56, height // 12)
                    result_margin = max(20, width // 40)
                    result_size = (120, 120)
                    result_gap = max(16, height // 40)
                    result_x = result_margin
                    result_y = banner_height + result_margin
                    annotated_frame = self.overlay_handshape(
                        annotated_frame, pred_right[0],
                        position=(result_x, result_y), size=result_size,
                    )
                    annotated_frame = self.overlay_handshape(
                        annotated_frame, pred_left[0],
                        position=(result_x, result_y + result_size[1] + 10 + result_gap),
                        size=result_size,
                    )

                    hidden_plot_size = (350, 150)
                    hidden_plot_right = self.plot_hidden_representation(self.saved_frames_hidrep_right.stack_and_get(), size=hidden_plot_size)
                    hidden_plot_left = self.plot_hidden_representation(self.saved_frames_hidrep_left.stack_and_get(), size=hidden_plot_size)
                    footer_height = max(34, height // 20)
                    plot_margin = max(20, width // 40)
                    plot_x = max(0, width - hidden_plot_size[0] - plot_margin)
                    plot_y_right = banner_height + plot_margin
                    plot_y_left = plot_y_right + hidden_plot_size[1] + plot_margin
                    if plot_y_left + hidden_plot_size[1] <= height - footer_height - plot_margin:
                        annotated_frame[
                            plot_y_right:plot_y_right + hidden_plot_size[1],
                            plot_x:plot_x + hidden_plot_size[0],
                        ] = hidden_plot_right
                        annotated_frame[
                            plot_y_left:plot_y_left + hidden_plot_size[1],
                            plot_x:plot_x + hidden_plot_size[0],
                        ] = hidden_plot_left

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

                    title = "HKU Linguistics: HKSL Handshape Detection Model Demonstration"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    title_scale = min(0.9, (width - 40) / max(
                        cv2.getTextSize(title, font, 1.0, 2)[0][0], 1
                    ))
                    title_size, _ = cv2.getTextSize(title, font, title_scale, 2)
                    title_x = max((width - title_size[0]) // 2, 10)
                    cv2.putText(
                        annotated_frame,
                        title,
                        (title_x, banner_height // 2 + title_size[1] // 2),
                        font,
                        title_scale,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    footer = (
                        f"[W] +10%  [E] -10%  Resolution: {resolution_percent}%  [Q] Quit"
                    )
                    footer_scale = 0.6
                    footer_size, _ = cv2.getTextSize(footer, font, footer_scale, 1)
                    cv2.putText(
                        annotated_frame,
                        footer,
                        ((width - footer_size[0]) // 2, height - footer_height // 2 + footer_size[1] // 2),
                        font,
                        footer_scale,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
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
                    cv2.imshow(window_name, render_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q')):
                        break
                    if key in (ord('w'), ord('W')):
                        resolution_percent = min(100, resolution_percent + 10)
                    elif key in (ord('e'), ord('E')):
                        resolution_percent = max(10, resolution_percent - 10)
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

def main():
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

    video_path = Path(__file__).resolve().parent / "B_01_079-NOTHING-0KC7-67.mp4"
    detector = HandDetection(model=model, video_path=video_path)
    detector.detect()

if __name__ == '__main__':
    main()
