import streamlit as st
from PIL import Image
from src.model import ImageClassificationService
import plotly.graph_objects as go

def get_emoji_for_label(label: str) -> str:
    label_lower = label.lower()

    animal_keywords = [
        'dog', 'cat', 'bird', 'fish', 'bear', 'lion', 'tiger', 
        'elephant', 'monkey', 'horse', 'cow', 'sheep', 'pig',
        'rabbit', 'fox', 'wolf', 'deer', 'zebra', 'giraffe',
        'panda', 'koala', 'kangaroo', 'penguin', 'owl'
    ]

    if any(keyword in label_lower for keyword in animal_keywords):
        return "🐾"

    food_keywords = [
        'food', 'pizza', 'burger', 'sandwich', 'hot dog', 'taco',
        'coffee', 'tea', 'juice', 'ice cream', 'cake', 'cookie',
        'bread', 'pasta', 'salad', 'soup', 'rice', 'noodle',
        'fruit', 'apple', 'banana', 'orange', 'strawberry'
    ]
    if any(keyword in label_lower for keyword in food_keywords):
        return '🍽️'  

    vehicle_keywords = [
        'car', 'truck', 'bus', 'vehicle', 'automobile',
        'airplane', 'aircraft', 'helicopter', 'train', 'subway',
        'boat', 'ship', 'motorcycle', 'bicycle', 'scooter'
    ]
    if any(keyword in label_lower for keyword in vehicle_keywords):
        return '🚗'  

    nature_keywords = [
        'tree', 'plant', 'flower', 'rose', 'grass', 'leaf',
        'mountain', 'forest', 'beach', 'ocean', 'river', 'lake'
    ]
    if any(keyword in label_lower for keyword in nature_keywords):
        return '🌿'  

    person_keywords = [
        'person', 'people', 'man', 'woman', 'child', 'boy', 'girl',
        'face', 'human'
    ]
    if any(keyword in label_lower for keyword in person_keywords):
        return '👤'  

    electronics_keywords = [
        'phone', 'computer', 'laptop', 'tablet', 'monitor', 'keyboard',
        'mouse', 'camera', 'television', 'tv', 'remote', 'headphone',
        'speaker', 'console', 'device', 'electronic'
    ]
    if any(keyword in label_lower for keyword in electronics_keywords):
        return '💻'  

    building_keywords = [
        'building', 'house', 'castle', 'church', 'temple', 'tower',
        'bridge', 'monument', 'stadium'
    ]
    if any(keyword in label_lower for keyword in building_keywords):
        return '🏢'  

    return '❓'  

def create_prediction_chart(predictions, top_k=5):
    top_predictions = predictions[:top_k]
    labels = [p['label'] for p in reversed(top_predictions)]
    scores = [p['score'] * 100 for p in reversed(top_predictions)]

    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=labels,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Blues',
                showscale=False
            ),
            text=[f'{s:.1f}%' for s in scores],
            textposition='auto'
        )
    ])

    fig.update_layout(
        title={
            'text': 'Top-5 Prediction',
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Confidence',
        yaxis_title='Predicted Label',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig

st.set_page_config(
    page_title="Image Classification",
    page_icon="📸",
    layout="wide"
)

st.title("AI Image Classification")
st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5f5f5;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.divider()

@st.cache_resource
def get_service():
    return ImageClassificationService()

with st.spinner("AI 모델 로딩 중..."):
    service = get_service()

st.subheader("이미지 입력 방식 선택")
tab1, tab2, tab3 = st.tabs(["파일 업로드", "카메라 촬영", "여러 이미지 선택"])

with tab1:
    st.write("이미지 파일을 업로드 하세요")
    uploaded_file = st.file_uploader(
        "이미지 선택",
        type=["jpg", "jpeg", "png"],
        key="file_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="업로드한 이미지", use_container_width=True)
        
        with col2:
            if st.button("분류하기", key='classify_upload'):
                with st.spinner("분류 중..."):
                    predictions = service.predict(image)
                    st.success("분석 완료")
                    
                    top1 = predictions[0]
                    emoji = get_emoji_for_label(top1['label'])
                    st.metric(
                        label=f"{emoji} 분석 결과",
                        value=top1['label'],
                        delta=f"{top1['score']*100:.1f}%"
                    )
                
                    st.divider()    

                    st.subheader("Top-5 예측 결과")
                    fig = create_prediction_chart(predictions)
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("카메라로 이미지 촬영")
    camera_photo = st.camera_input("촬영")

    if camera_photo is not None:
        image = Image.open(camera_photo)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="촬영한 이미지", use_container_width=True)

        with col2:
            if st.button("분류하기", key="classify_camera"):
                with st.spinner("분류 중..."):
                    predictions = service.predict(image)
                    st.success("분석 완료")

                    top1 = predictions[0]
                    emoji = get_emoji_for_label(top1['label'])
                    st.metric(
                        label=f"{emoji} 분석 결과",
                        value=top1['label'],
                        delta=f"{top1['score']*100:.1f}%"
                    )

                    st.divider()
                    st.subheader("Top-5 예측 결과")
                    fig = create_prediction_chart(predictions)
                    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.write("여러 이미지 선택 후 분류")
    uploaded_files = st.file_uploader(
        "이미지들 선택",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if uploaded_files:
        st.info(f"총 {len(uploaded_files)}개의 이미지가 선택되었습니다")
        if st.button("모두 분류하기", key="classify_batch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for idx, file in enumerate(uploaded_files):
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"처리 중 {idx + 1}/{len(uploaded_files)}")
                image = Image.open(file)
                predictions = service.predict(image)
                results.append({
                    "file_name": file.name,
                    "image": image,
                    "predictions": predictions
                })

            status_text.text("모든 이미지 처리 완료")
            progress_bar.empty()
            st.divider()
            st.subheader("배치 처리 결과")

            for result in results:
                top1 = result['predictions'][0]
                emoji = get_emoji_for_label(top1['label'])
                with st.expander(
                        f"{emoji} {result['file_name']} → {top1['label']} ({top1['score']*100:.1f}%)"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(result['image'], use_container_width=True)

                    with col2:
                        fig = create_prediction_chart(result['predictions'])
                        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("미션 17 6팀 이승완")