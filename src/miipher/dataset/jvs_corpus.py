import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import MeCab


class JVSCorpus(Dataset):
    def __init__(self, root, exclude_speakers=[]) -> None:
        super().__init__()
        self.root = Path(root)
        self.speakers = [
            f.stem
            for f in self.root.glob("jvs*")
            if f.is_dir() and f.stem not in exclude_speakers
        ]
        self.clean_texts = dict()
        self.wav_files = []
        for speaker in self.speakers:
            transcript_files = (self.root / speaker).glob("**/transcripts_utf8.txt")
            for transcript_file in transcript_files:
                subset = transcript_file.parent.name
                with transcript_file.open() as f:
                    lines = f.readlines()
                for line in lines:
                    wav_name, text = line.strip().split(":")
                    self.clean_texts[f"{speaker}/{subset}/{wav_name}"] = text
                    wav_path = self.root / Path(
                        f"{speaker}/{subset}/wav24kHz16bit/{wav_name}.wav"
                    )
                    if wav_path.exists():
                        self.wav_files.append(wav_path)

        self.tokenizer = MeCab.Tagger("-Owakati")

    def __getitem__(self, index):
        wav_path = self.wav_files[index]
        wav_tensor, sr = torchaudio.load(wav_path)
        wav_path = wav_path.resolve()
        speaker = wav_path.parent.parent.parent.stem
        subset = wav_path.parent.parent.stem
        wav_name = wav_path.stem

        clean_text = self.clean_texts[f"{speaker}/{subset}/{wav_name}"]

        basename = f"{subset}_{speaker}_{wav_name}"
        tokenized = self.tokenizer.parse(clean_text)
        output = {
            "wav_path": str(wav_path),
            "speaker": speaker,
            "clean_text": clean_text,
            "word_segmented_text": tokenized,
            "basename": basename,
            "lang_code": "jpn",
        }

        return output

    def __len__(self):
        return len(self.wav_files)

    @property
    def speaker_dict(self):
        speakers = set()
        for wav_path in self.wav_files:
            speakers.add(wav_path.parent.parent.parent.stem)
        speaker_dict = {x: idx for idx, x in enumerate(speakers)}
        return speaker_dict

    @property
    def lang_code(self):
        return "jpn"
