# 🧠 OpenVINO LLM 변환 및 추론

 PyTorch의 LLM 모델을 OpenVINO 형식으로 변환하고, 양자화를 적용한 후, 추론 성능을 측정

## 📦 환경 설정

체크포인트 디렉토리 만들기
```sh
mkdir ckpts/ 
```

가상환경 세팅
```sh
python3 -m venv venv
source venv/bin/activate  # Windows는 venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🔄 모델 변환 및 양자화

### 기존 Huggingface 모델 추론 돌려보기
```sh
python original_inference.py
```

### LLM 모델을 OpenVINO IR 형식으로 변환

```sh
python convert.py
```


## 🚀 추론 및 성능 측정

```sh
python inference.py
```

- PyTorch, OpenVINO, 양자화 모델 각각에 대해 추론을 실행하고 성능을 비교
- 출력: 추론 시간, 예측 결과 등

--- 

## 참조
- [Optimum LLM Conversion](https://docs.openvino.ai/2025/openvino-workflow-generative/inference-with-optimum-intel.html)
