# capstone1
종합설계프로젝트1 - EasyOCR 로 한국어 손글씨 인식 향상시키기\

교수평가 일치 및 교수학습 효율성 증대를 위한 에듀테크 도구 \
개발/연구 활동 기간: 2023.04 ~ 2023.06 종합설계 프로젝트 수업의 산학협력 과제였습니다. \
손글씨 교육을 위한 에듀테크 도구 개발 및 기술 연구로써 OCR을 활용한 한글 인식과 맞춤법 검사를 제공하는 어플리케이션을 개발하였습니다. \
해당 어플리케이션은 구글 플레이스토어에 런칭하는 성과를 보였습니다. 에듀테크 도구 개발에 대해 논문을 작성하여 KCC2023 (한국컴퓨터종합학술대회)에 발표 및 게재하였습니다.\ 
해당 논문은 KCC2023 학부생 부문 장려상을 수상하였습니다. \
참고 링크: 최종 보고서 링크[https://docs.google.com/document/d/1UH6kkQHILb8OkAq7K7DwuT1pGV57Ogoi/edit]

------
# Colab 폴더 : 

### GPU_EASYOCR_ENV
cuda, 파이썬 환경 설정 
### Tesseract_OCR_SampleData.ipynb
테서렉트로 한국어 학습 후 한국어데이터(AI Hub 이용) 인식률 확인해봄.
### compare_Tesseract_vs_EasyOCR 
테서렉트와 easyocr의 성능 비교
### EASY_OCR_train.ipynb
easyOcr 한국어모델 koreaG2모델을 이용해 한국어 단어(위키디피아 모듈을 통해 생성)학습시킨 후, 성능 확인
### Aihub_data_prepare.ipynb
모델에 학습시키기 위해 데이터를 학습용 데이터에 맞게 파일명 변경하는 코드
### (Latest)Model_training.ipynb
데이터 전처리 후 deep-text-recognition-benckmark 모델 학습하는 코드

# data 
[Ai-Hub 한국어 글자체 이미지](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=81)
# prepare
------

      git clone https://github.com/JaidedAI/EasyOCR.git
      git clone https://github.com/clovaai/deep-text-recognition-benchmark.git
      
------

# train :
------
### EASYocr 추가학습
            CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./deep-text-recognition-benchmark/train.py \
             --train_data ./deep-text-recognition-benchmark/htr_data_lmdb/train \
             --valid_data ./deep-text-recognition-benchmark/htr_data_lmdb/validation \
             --Transformation None \
            --FeatureExtraction VGG \
            --SequenceModeling BiLSTM \
            --Prediction CTC \
            --input_channel 1 \
            --output_channel 256 \
            --hidden_size 256 \
            --batch_max_length 50 \
            --batch_size 1\
             --saved_model ./pretrained_models/korean_g2.pth \
            --lr 0.1 \
             --character " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~가각간갇갈감갑값강갖같갚갛개객걀거걱건걷걸검겁것겉게겨격겪견결겹경곁계고곡곤곧골곰곱곳공과관광괜괴굉교구국군굳굴굵굶굽궁권귀규균그극근글긁금급긋긍기긴길김깅깊까깎깐깔깜깝깥깨꺼꺾껍껏껑께껴꼬꼭꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내냄냉냐냥너넉널넓넘넣네넥넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨누눈눕뉘뉴늄느늑는늘늙능늦늬니닐님다닥닦단닫달닭닮담답닷당닿대댁댐더덕던덜덤덥덧덩덮데델도독돈돌돕동돼되된두둑둘둠둡둥뒤뒷드득든듣들듬듭듯등디딩딪따딱딴딸땀땅때땜떠떡떤떨떻떼또똑뚜뚫뚱뛰뜨뜩뜯뜰뜻띄라락란람랍랑랗래랜램랫략량러럭런럴럼럽럿렁렇레렉렌려력련렬렵령례로록론롬롭롯료루룩룹룻뤄류륙률륭르른름릇릎리릭린림립릿마막만많말맑맘맙맛망맞맡맣매맥맨맵맺머먹먼멀멈멋멍멎메멘멩며면멸명몇모목몰몸몹못몽묘무묵묶문묻물뭄뭇뭐뭣므미민믿밀밉밌및밑바박밖반받발밝밟밤밥방밭배백뱀뱃뱉버번벌범법벗베벤벼벽변별볍병볕보복볶본볼봄봇봉뵈뵙부북분불붉붐붓붕붙뷰브블비빌빗빚빛빠빨빵빼뺨뻐뻔뻗뼈뽑뿌뿐쁘쁨사삭산살삶삼상새색샌생서석섞선설섬섭섯성세센셈셋션소속손솔솜솟송솥쇄쇠쇼수숙순술숨숫숲쉬쉽슈스슨슬슴습슷승시식신싣실싫심십싱싶싸싹쌀쌍쌓써썩썰썹쎄쏘쏟쑤쓰쓸씀씌씨씩씬씹씻아악안앉않알앓암압앗앙앞애액야약얇양얗얘어억언얹얻얼엄업없엇엉엌엎에엔엘여역연열엷염엽엿영옆예옛오옥온올옮옳옷와완왕왜왠외왼요욕용우욱운울움웃웅워원월웨웬위윗유육율으윽은을음응의이익인일읽잃임입잇있잊잎자작잔잖잘잠잡장잦재쟁저적전절젊점접젓정젖제젠젯져조족존졸좀좁종좋좌죄주죽준줄줌줍중쥐즈즉즌즐즘증지직진질짐집짓징짙짚짜짝짧째쨌쩌쩍쩐쪽쫓쭈쭉찌찍찢차착찬찮찰참창찾채책챔챙처척천철첫청체쳐초촉촌총촬최추축춘출춤춥춧충취츠측츰층치칙친칠침칭카칸칼캐캠커컨컬컴컵컷켓켜코콜콤콩쾌쿠퀴크큰클큼키킬타탁탄탈탑탓탕태택탤터턱털텅테텍텔템토톤톱통퇴투툼퉁튀튜트특튼튿틀틈티틱팀팅파팎판팔패팩팬퍼퍽페펴편펼평폐포폭표푸푹풀품풍퓨프플픔피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈협형혜호혹혼홀홍화확환활황회획횟효후훈훌훔훨휘휴흉흐흑흔흘흙흡흥흩희흰히힘" \
                 --FT


### 처음부터 학습
            CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./deep-text-recognition-benchmark/train.py \
                --train_data ./deep-text-recognition-benchmark/htr_data_lmdb/train \
                --valid_data ./deep-text-recognition-benchmark/htr_data_lmdb/validation \
                --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC \   —batch_max_length 50 \
            --batch_size 1 


------

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
