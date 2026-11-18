"""Compose actual match visualizations and trained Gaussian frames; CPU only."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = Path('/mnt/afs/260010098/code/3dv_assignment')
ROOT = Path('/mnt/afs/260010098/data/3dv_assignment/roma100_filter_r01')
REPORT = ROOT / 'assignment2_report'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ShowcaseMontage:
    gap = 16
    matching_width = 640
    render_size = 320
    header = 80
    row_height = 368
    background = '#edf2f6'

    def __init__(self):
        self.items = json.loads((REPORT / 'artifact_manifest.json').read_text())['showcase']
        assert len(self.items) == 10
        self.width = 5 * self.gap + 2 * (self.matching_width + self.render_size)
        self.height = self.header + 5 * self.row_height + self.gap
        self.font = ImageFont.load_default(size=20)
        self.small = ImageFont.load_default(size=16)
        self.title_font = ImageFont.load_default(size=26)
        self.base = Image.new('RGB', (self.width, self.height), self.background)
        draw = ImageDraw.Draw(self.base)
        draw.text((self.gap, 10), 'RoMa v2 + foreground / confidence filtering | SfM | 3DGS', fill='#173e4d', font=self.title_font)
        self.placements = []
        self.evidence = []
        for side in range(2):
            x = self.gap + side * (self.matching_width + self.render_size + 2 * self.gap)
            draw.text((x, 50), 'POINT CORRESPONDENCES', fill='#58717e', font=self.small)
            draw.text((x + self.matching_width + self.gap, 50), '3DGS RENDERING', fill='#58717e', font=self.small)
        for i, item in enumerate(self.items):
            x = self.gap + (i % 2) * (self.matching_width + self.render_size + 2 * self.gap)
            y = self.header + (i // 2) * self.row_height
            label = f"{i + 1:02d}  {item['name']}  [{item['uid'][:8]}]"
            draw.text((x, y), label, fill='#173e4d', font=self.font)
            y += 32
            matching = REPORT / item['match']
            with Image.open(matching) as im:
                panel = ImageOps.pad(im.convert('RGB'), (self.matching_width, self.render_size),
                                     method=Image.Resampling.LANCZOS, color='white')
            self.base.paste(panel, (x, y))
            render_xy = (x + self.matching_width + self.gap, y)
            source = ROOT / 'results/fg_conf050' / item['uid'] / 'gaussian/centered'
            frames = sorted(source.glob('frame_*.png'))
            expected = json.loads((REPORT / 'assets/showcase' / item['uid'] / 'gaussian/png_sequence.json').read_text())['frames']
            assert len(frames) == len(expected) == 72
            for path, record in zip(frames, expected):
                assert path.name == record['name'] and digest(path) == record['sha256']
            self.placements.append((render_xy, frames))
            self.evidence.append({'uid': item['uid'], 'name': item['name'], 'row': i // 2 + 1,
                                  'columns': [1, 2] if i % 2 == 0 else [3, 4],
                                  'match_source': str(matching), 'match_sha256': digest(matching),
                                  'render_source': str(source), 'frames': expected})

    def frame(self, index: int) -> Image.Image:
        canvas = self.base.copy()
        for xy, paths in self.placements:
            with Image.open(paths[index]) as im:
                assert im.size == (512, 512)
                canvas.paste(im.convert('RGB').resize((self.render_size, self.render_size), Image.Resampling.LANCZOS), xy)
        return canvas

    def save(self):
        output = REPORT / 'showcase_overview.gif'
        # One palette for every frame keeps static matches and labels stable.
        thumb = (self.width // 4, self.height // 4)
        palette_source = Image.new('RGB', (thumb[0] * 2, thumb[1] * 2))
        for i, index in enumerate([0, 18, 36, 54]):
            preview = self.frame(index)
            if index == 0:
                preview.save(REPORT / 'showcase_overview.png')
            palette_source.paste(preview.resize(thumb, Image.Resampling.LANCZOS),
                                 ((i % 2) * thumb[0], (i // 2) * thumb[1]))
        palette = palette_source.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        frames = []
        for index in range(72):
            frames.append(self.frame(index).quantize(palette=palette, dither=Image.Dither.NONE))
            if index % 18 == 0:
                print(f'Composed {index + 1}/72 frames', flush=True)
        durations = [(round((i + 1) * 100 / 24) - round(i * 100 / 24)) * 10 for i in range(72)]
        temporary = output.with_suffix('.pending.gif')
        frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, disposal=1, optimize=False)
        with Image.open(temporary) as check:
            total = 0
            for i in range(check.n_frames):
                check.seek(i)
                total += check.info['duration']
            assert total == 3000 and check.size == (self.width, self.height)
            encoded_frames = check.n_frames
        temporary.replace(output)
        metadata = {'layout': '5 rows; matching, rendering, matching, rendering per row',
                    'size': [self.width, self.height], 'source_frames': 72,
                    'encoded_frames': encoded_frames, 'duration_ms': 3000, 'fps': 24,
                    'display_only': True, 'preprocessing_changed': False,
                    'display_resize': 'Matching 2560x1280 -> 640x320; render 512x512 -> 320x320; full frame, no cropping.',
                    'encoding': 'Shared 256-color GIF palette, no dithering, 10ms duration quantization.',
                    'sources': self.evidence, 'sha256': digest(output), 'bytes': output.stat().st_size}
        sidecar = output.with_suffix('.json')
        sidecar.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        shutil.copyfile(output, REPO / 'assets/showcase_overview.gif')
        shutil.copyfile(sidecar, REPO / 'assets/showcase_overview.json')
        print(json.dumps({k: metadata[k] for k in ['size', 'encoded_frames', 'duration_ms', 'bytes']}) + '\n' + str(output))


if __name__ == '__main__':
    ShowcaseMontage().save()
