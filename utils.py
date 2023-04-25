from torch.utils.data import Dataset, DataLoader
import pretty_midi
import pandas as pd
import os
import torch

class MIO:
   def __init__(self, dataset_path='groove'):
    self.dataset_path = dataset_path

   def load(self, music_id):
    self.midi = pretty_midi.PrettyMIDI(os.path.join(self.dataset_path, music_id))
    self.notes = [[note.start, note.end, note.pitch, note.velocity] for note in self.midi.instruments[0].notes]
    return self

   def extract_unit_notes(self, time_signature, bpm):
    unit_duration = 60*int(time_signature[0])/bpm
    self.unit_notes = torch.tensor([[note.start, note.end, note.pitch, note.velocity] for note in self.notes if note[0] <= unit_duration])
    return self

class MidiDataset(Dataset):
    def __init__(self, dataset_path, opt='train'):
        
        self.data = []
        self.dataset_path = dataset_path
        df = pd.read_csv(os.path.join(dataset_path, 'info.csv'))
        df = df.loc[df.split == opt, :]
        mio = MIO()
        self.data = df.apply(lambda row: mio.load(row.id).extract_unit_notes(map(int, row.time_signature.split('-')), row.bpm, row.duration).unit_notes, axis =1)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]


def load_data(dataset_path, num_workers=0, batch_size=512):
    dataset = MidiDataset(dataset_path, opt='train')
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)