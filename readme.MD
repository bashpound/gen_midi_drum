## 활용 프레임워크/패키지
- learning: pytorch
- io: pretty_midi
- data processing: pandas, numpy

## 데이터 전처리(utils.py)

1. pretty_midi package를 이용해 info.csv내 midi sequence note를 추출
2. 학습속도 및 성능을 위해 info.csv내 bpm과 time signature를 통해 1마디에 해당하는 duration을 찾은 후, 4마디까지만 컷
3. 4마디의 sequence length가 되지 않는 데이터는 [start, end, pitch, velocity] = [0,0,0,0] 으로 padding
4. 학습을 위해 torch.utils.data.DataSet를 상속하는 dataset 구현

## 학습(train.py)

1. batch size는 instance 스펙을 고려, 32-256로 진행 (논문=512)
2. (ISSUE) kl loss만 학습되고, reconstruction loss는 학습되지 않는 문제

## 생성(utils.make_midi)

1. 모델 output shape는 [batch_size, sequence_length, 4]이므로, [sequence_length, 4]에 해당하는 [start, end, pitch, velocity] 노트를 직접 생성
2. 해당 노트를 직접 midi로 write 하는 함수 작성


## 보완점 (미진행)

1. 데이터 처리부에서 start, end, pitch, velocity를 직접 투입하지 않고, 적절한 형태로 가공하여 학습이 필요
2. output에서 note를 직접 생성하지 않고 track을 잡아서 실제 scalar value를 도출하는 부분(생성부) 구현이 필요
3. Drum 의 pitch가 악기를 결정한다는 사실을 뒤늦게 알아서 데이터 처리를 다시 해야 올바른 학습이 될 것 같습니다.

