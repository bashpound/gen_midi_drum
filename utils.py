from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pretty_midi
import pandas as pd
from torchvision import transforms



def extract_notes(path):
    m = pretty_midi.PrettyMIDI(path)
    return [[note.start, note.end, note.pitch, note.velocity] for note in m.instruments[0].notes]


class MidiDataset(Dataset):
    def __init__(self, dataset_path, train_test_val='train'):
        import os

        self.data = []
        to_tensor = transforms.ToTensor()
        df = pd.read_csv(os.path.join(dataset_path, 'info.csv'))
        df = df.loc[df.split == train_test_val, :]
        for mid_path, label in zip(df.midi_filename, df.style):
            notes = extract_notes(os.path.join(dataset_path, mid_path))
            self.data.append((to_tensor(notes), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=512):
    dataset = MidiDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)

