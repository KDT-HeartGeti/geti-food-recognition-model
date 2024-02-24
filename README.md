# Food Recognition Model using Intel's Geti Platform (인텔 컴퓨터비전 플랫폼 Geti를 이용한 음식 인식 모델)
1. Geti로 만든 모델
2. 음식 인식 모델을 Flask 웹 앱으로 배포

## 실행 메뉴얼
1. https://github.com/KDT-HeartGeti/Geti를 안드로이드 스튜디오로 clone합니다.
2. https://github.com/KDT-HeartGeti/geti-food-recognition-model를 Visual Studio Code로 Clone합니다.
3. [구글 드라이브](https://drive.google.com/drive/folders/1-SEJvPtaYeIHxsEJ2S2TBn_iXO1DTQu1?usp=sharing)에서 모델 파일을 다운받습니다.
4. flasktflite.py를 실행합니다.
5. cmd에서 ipconfig 쳐서 나온  IPv4 주소를 복사 하기 (192로 시작)
6. 복사한 주소 파이썬 파일에 있는 주소와 바꾸기
7. 서버 실행
8. 안드로이드 스튜디오에서 LoadingScreen에서 서버 주소 바꾸기
    복사한 주소/prediction 까지 쳐야함
9. 안드로이드 스튜디오에서 실행

## 인텔 게티 플랫폼 : 인텔의 컴퓨터 비전 인공지능 플랫폼입니다.
### 음식 이미지 Annotation 하는 과정 스크린샷
| [The Intel® Geti™ Platform - Intel's Computer Vision AI Platform](https://geti.intel.com/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="800" alt="스크린샷 2024-01-17 오후 2 20 58" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/058133ba-9e8e-4869-8fad-d539f4cb7e79"> <img width="800" alt="스크린샷 2024-01-17 오후 2 21 17" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/46c4a1a7-ed82-4feb-8a67-c19d2f2ba5d1"> <img width="800" alt="스크린샷 2024-01-17 오후 2 21 22" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/601a0e8a-e072-42c0-93c2-b746c1b9e356"> <img width="800" alt="스크린샷 2024-01-17 오후 2 21 28" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/df152b95-7147-466e-ac3f-21a9da3a08ec"> <img width="800" alt="스크린샷 2024-01-17 오후 2 21 37" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/7052d7db-197f-40ba-83e1-bee87dac4399"> <img width="800" alt="스크린샷 2024-01-17 오후 2 21 52" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/f838e6f3-4972-4c3a-af3a-ce18881e6353"> |

## 앱 스크린샷
| [초기화면]                                                                                                                                                | [내 상태]                                                                                                                                                 | [내상태]                                                                                                                                                  | [영양정보]                                                                                                                                                |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="200" alt="스크린샷 2024-01-17 오후 2 22 35" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/c20ef633-396a-41de-a3bd-bcff20ef15b4"> | <img width="200" alt="스크린샷 2024-01-17 오후 2 22 41" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/fc4110f1-7c02-4ecb-8299-9d7b7c49064a"> | <img width="200" alt="스크린샷 2024-01-17 오후 2 22 47" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/4f1ee8c3-f6a7-49a3-8cb6-5866c6ee11c3"> | <img width="200" alt="스크린샷 2024-01-17 오후 2 22 55" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/370ff5c0-cbbb-47de-aeff-efdb8bf411c6"> |

| [사진찍어서 영양정보 분석하기]                                                                                                                            | [사진찍어서 영양정보 분석하기]                                                                                                                            | [갤러리에서 영양정보 분석하기]                                                                                                              | [건강기능식품 광고 화면]                                                                                                                                  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="200" alt="스크린샷 2024-01-17 오후 2 23 06" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/0d27f2a1-e5bd-4e9d-8e78-70a0df4d2149"> | <img width="200" alt="스크린샷 2024-01-17 오후 2 23 14" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/178e03cb-f805-4490-bf77-78323c1cefa8"> | <img width="200" alt="스크린샷 2024-01-17 오후 2 23 28" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/e6569b45-51ea-4d6f-91d9-578b6dbb2f0a"> | <img width="200" alt="스크린샷 2024-01-17 오후 2 23 39" src="https://github.com/KDT-HeartGeti/Geti/assets/71699054/bbf14226-1844-4edb-909d-079a74360380"> |



## Thanks [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FKDT-HeartGeti%2FGeti&count_bg=%2345FFCA&title_bg=%23FFB6D9&icon=&icon_color=%23E7E7E7&title=Heart_Geti&edge_flat=false)](https://hits.seeyoufarm.com)
- 참여해주신 모든 분들 감사합니다!
- GitHub : [Contributors](https://github.com/KDT-HeartGeti/Geti/pulse)
- Designer : SangEun Kim
- Maintainer : [vmkmym](https://github.com/vmkmym), [21dbwls12](https://github.com/21dbwls12)
