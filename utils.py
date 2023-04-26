from torch.utils.data import Dataset, DataLoader
import pretty_midi
import pandas as pd
import os
import torch

#midi IO module
class MIO:
   def __init__(self, dataset_path='groove'):
    self.dataset_path = dataset_path

   def load(self, midi_filename):
    self.midi = pretty_midi.PrettyMIDI(os.path.join(self.dataset_path, midi_filename))
    self.notes = [[note.start, note.end, note.pitch, note.velocity] for note in self.midi.instruments[0].notes]
    return self

   def extract_unit_notes(self, time_signature, bpm):
    unit_duration = 4*60*int(time_signature[0])/bpm
    self.unit_notes = torch.tensor([note for note in self.notes if note[0] <= unit_duration][:32])
    return self


#dataset
class MidiDataset(Dataset):
    def __init__(self, dataset_path, opt='train'):
        
        self.data = []
        self.dataset_path = dataset_path

        #csv load
        df = pd.read_csv(os.path.join(dataset_path, 'info.csv'))
        df = df.loc[df.split == opt, :]

        mio = MIO()

        #각 bpm과 time signature별로 4마디 안에 해당하는 note sequence를 선별
        data = df.apply(lambda row: mio.load(row.midi_filename).extract_unit_notes(tuple(map(int, row.time_signature.split('-'))), row.bpm).unit_notes, axis =1)
        
        #Tensor sequence note padding (너무 짧은 sequence의 경우 나머지를 0으로 채움)
        self.data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0).to(torch.float32)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#sequence note로부터 midi를 생성하는 함수
def make_midi(prob_matrix, fpath):
  midi_data = pretty_midi.PrettyMIDI()

  #output tensor를 binary 확률분포로 가정하고 이진화
  sequence_note = torch.distributions.Bernoulli(prob_matrix).sample().detach().numpy()
  

  instrument = pretty_midi.Instrument(program=0, is_drum=True) 
  interval = 8/64

  for idx, (start, end, pitch, velocity) in enumerate(sequence_note):
    idx = idx*interval
    if start == 1:
       start = idx
       end = start+0.1
       pitch = 38 if pitch == 1 else 36
       velocity = 100 if velocity == 1 else 50
       
       note = pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=start, end=end)
       instrument.notes.append(note)

  midi_data.instruments.append(instrument)
  midi_data.write(fpath)


def load_data(dataset_path, num_workers=0, batch_size=512, opt='train'):
    dataset = MidiDataset(dataset_path, opt=opt)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)