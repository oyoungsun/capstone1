# OCRproject
# capstone1
종합설계프로젝트 - EasyOCR 로 한국어 손글씨 인식 향상시키기


### Tesseract_OCR_SampleData.ipynb
테서렉트로 한국어 학습 후 한국어데이터(AI Hub 이용) 인식률 확인해봄.
### compare_Tesseract_vs_EasyOCR 
테서렉트와 easyocr의 성능 비교

### EASY_OCR_train.ipynb
easyOcr 한국어모델 koreaG2모델을 이용해 한국어 단어(위키디피아 모듈을 통해 생성)학습시킨 후, 성능 확인

### Aihub_data_prepare.ipynb
모델에 학습시키기 위해 데이터를 학습용 데이터에 맞게 파일명 변경하는 코드


# deep-text-recognition-benckmark 기반 EasyOCR 추가학습     
    deep-text-recognition-benchmark
      |--htr_data              
        |--train              # AIhub 손글씨
        |--test               
        |--validation        
        |--get_images.py      # kor_dataset/aihub_data/htr/images에서 이미지 파일들을 가져와 train, test, validation 폴더로 분리
        |--gt_train.py        # train dataset gt file
        |--gt_test.py         # test dataset gt file
        |--gt_validation.py   # validation dataset gt file
      |--demo.py              # pretrained model을 test               
      |--train.py             # model training
      |--test.py
      |--...
        
    kor_dataset
      |--aihub_data
        |--htr
          |--images                       # AIhub 필기체 이미지 파일들을 포함
          |--handwriting_data_info1.json  # AIhub 필기체 이미지들에 대한 라벨링 파일
      |--finetuning_data
        |--made1
           |--images                       # 직접 제작한 데이터셋 이미지 파일들을 포함
           |--labels.txt                   # 직접 제작한 데이터셋 이미지들에 대한 라벨링 파일
           
    saved_models                           # 학습 완료 된 모델 저장위
      |--None-ResNet-None-Attn-Seed1111 
        |--best_accuracy.pth               # 정확도 제일 높은 pretrained model
        |--...
        
    pretrained_models                      # EasyOCr_korean_g2 모델 넣어두는 곳
      |--kocrnn.pth
      |--korean_g2.pth
        
    test                                   # 별도로 테스트할 dataset들을 저장하는 폴더
      |--images
      |--labels.txt
