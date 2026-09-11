import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# audio/scripts/<berkas ini> -> root proyek ada di parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from common import EMOTIONS, make_audio_cnn, load_wav_mfcc

# ── KONFIGURASI ──────────────────────────────
# EMOTIONS dan arsitektur model diimpor dari common.py
# supaya arsitekturnya tidak pernah berbeda dengan skrip yang memuat bobotnya.
DATA_DIR = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"
EPOCHS = 30
BATCH = 32
DEVICE = 'cpu'      # ganti 0 / 'cuda' kalau ada GPU
# ─────────────────────────────────────────────

class AudioDataset(Dataset):
    def __init__(self, split):
        self.samples = []
        for idx, emo in enumerate(EMOTIONS):
            folder = DATA_DIR / split / emo
            if folder.exists():
                for f in folder.glob("*.wav"):
                    self.samples.append((f, idx))
        print(f"{split}: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        path, label = self.samples[i]
        mfcc = load_wav_mfcc(path)
        return torch.tensor(mfcc).unsqueeze(0), label  # (1, 40, 94)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

if __name__ == '__main__':
    train_ds = AudioDataset("train")
    val_ds = AudioDataset("val")
    
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=0)
    
    model = make_audio_cnn(len(EMOTIONS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    save_dir = PROJECT_ROOT / "audio" / "models"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        val_acc = evaluate(model, val_dl)
        print(f"Epoch {epoch}/{EPOCHS} | loss: {total_loss/len(train_dl):.4f} | val_acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_dir / "best_audio_cnn.pt")
    
    print(f"\nTraining selesai! Best val accuracy: {best_acc:.4f}")
    print(f"Model disimpan di: {save_dir / 'best_audio_cnn.pt'}")