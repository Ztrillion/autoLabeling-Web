# autoLabeling-Web

## 231004 
### UI
* 이미지 로드
* 오토 라벨링 시각화
* BBOX 및 BBOX좌표(XYXY, XYWHN),신뢰도,클래스 이름 표로 출력
* 원하는 객체만 hilight
* 저품질 객체 삭제

### BACK
* AI 로드 및 이미지 데이터 전처리 후 예측
* 예측된 BBOX 중 예측하지 않아도 되는 클래스를 제외 후 출력
