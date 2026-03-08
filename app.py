import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 페이지 설정
st.set_page_config(page_title="국내 주가 예측 AI", layout="wide")

st.title("🇰🇷 AI 국내 주가 예측 서비스")
st.markdown("종목 코드를 입력하면 AI가 향후 14일간의 추세를 예측합니다.")

# 사용자 입력
with st.sidebar:
    st.header("설정")
    stock_code = st.text_input("종목 코드 6자리 (예: 삼성전자 005930)", value="005930")
    analyze_btn = st.button("분석 시작")

if analyze_btn:
    try:
        with st.spinner('데이터를 불러오는 중...'):
            # 1. 데이터 불러오기 (최근 1년치)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            df = fdr.DataReader(stock_code, start_date, end_date)

            if df.empty:
                st.error("종목 코드를 확인해주세요. (숫자 6자리)")
            else:
                # 2. AI 예측 모델 (안정적인 통계 모델 사용)
                model = ExponentialSmoothing(df['Close'], trend='add', seasonal=None).fit()
                forecast = model.forecast(14) # 14일 예측
                
                # 날짜 생성
                last_date = df.index[-1]
                forecast_dates = [last_date + timedelta(days=i) for i in range(1, 15)]

                # 3. 차트 시각화
                fig = go.Figure()
                # 과거 주가 라인
                fig.add_trace(go.Scatter(x=df.index[-90:], y=df['Close'].tail(90), name='과거 주가', line=dict(color='blue')))
                # 예측 주가 라인
                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, name='AI 예측값', line=dict(dash='dash', color='red')))
                
                fig.update_layout(title=f"[{stock_code}] 주가 추이 및 예측", xaxis_title="날짜", yaxis_title="가격(원)")
                st.plotly_chart(fig, use_container_width=True)

                # 4. 결과 정보 제공
                curr_price = df['Close'].iloc[-1]
                pred_price = forecast.iloc[-1]
                diff = pred_price - curr_price
                
                col1, col2, col3 = st.columns(3)
                col1.metric("현재가", f"{curr_price:,.0f}원")
                col2.metric("14일 뒤 예상가", f"{pred_price:,.0f}원")
                col3.metric("예상 변동폭", f"{diff:,.0f}원", delta=f"{diff:,.0f}원")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
