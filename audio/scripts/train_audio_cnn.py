import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── KONFIGURASI ──────────────────────────────
# audio/scripts/<this file> -> project root is parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "audio" / "datasets" / "audio_emotion"
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SR = 16000          # sampling rate (sesuai proposal: 16 kHz)
DURATION = 3        # detik (dipotong/dipad ke 3 detik)
N_MFCC = 40         # jumlah koefisien MFCC
EPOCHS = 30
BATCH = 32
DEVICE = 'cpu'      # ganti 0 / 'cuda' kalau ada GPU
# ─────────────────────────────────────────────

def extract_mfcc(path):
    """Ekstrak MFCC dari file audio, hasil shape konsisten"""
    y, sr = librosa.load(path, sr=SR)
    target_len = SR * DURATION
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    return mfcc.astype(np.float32)   # shape: (40, 94)

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
        mfcc = extract_mfcc(path)
        return torch.tensor(mfcc).unsqueeze(0), label  # (1, 40, 94)

class AudioCNN(nn.Module):
    def __init__(self, n_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        return self.net(x)

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
    
    model = AudioCNN(len(EMOTIONS)).to(DEVICE)
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