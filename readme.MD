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

1. batch size는 instance 스펙을 고려, 32로 진행 (논문 512)
2. (ISSUE) kl loss만 학습되고, reconstruction loss는 학습되지 않는 문제
3. (ISSUE) 2의 reconstruction loss에서 기인한 gradient explode 문제
    - Dataset을 start, end, pitch, velocity 4 channel의 시퀀스 길이를 갖도록 transpose로 학습
    - CrossEntropy > MSELoss로 변경, activation layer 제거
    - 모델 혹은 reconstruction loss에 문제가 있는 것 같은데, 아직 해결하지 못했습니다.
    - model output이 sequence note를 직접 만드는 방식이 문제가 있음을 인지하였지만 시간상 해결하지 못했습니다.

## 생성(utils.make_midi)

1. 모델 output shape는 [batch_size, sequence_length, 4]이므로, [sequence_length, 4]에 해당하는 [start, end, pitch, velocity] 노트를 직접 생성
2. 해당 노트를 직접 midi로 write 하는 함수 작성