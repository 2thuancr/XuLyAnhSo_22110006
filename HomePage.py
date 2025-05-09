import streamlit as st
#import lib.common as tools

st.set_page_config(
    page_title="Đồ án cuối kỳ",
)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-040.jpg");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("");
    background-position: center;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-040.jpg");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)


st.write("# Đồ án cuối kỳ")

st.markdown(
    """
    ## Sản phẩm
    Project cuối kỳ cho môn học xử lý ảnh số DIPR430685_23_2_03.
    Thuộc Trường Đại Học Sư Phạm Kỹ Thuật TP.HCM.
    ### 9 chức năng chính trong bài
    - 📖Giải phương trình bậc 2
    - 📖Nhận dạng khuôn mặt
    - 📖Nhận dạng cử chỉ 
    - 📖Nhận dạng đối tượng yolo4 dùng sample của opencv
    - 📖Nhận dạng chữ viết tay MNIST
    - 📖Nhận dạng 5 loại trái cây
    - 📖Xử lý ảnh số
    - 📖Nhận dạng màu sắc
    - 📖Nhận dạng phương tiện giao thông và đếm số lượng phương tiện.
    ## Thông tin sinh viên thực hiện
    - Họ tên: Vi Quốc Thuận
    - MSSV: 22110006
    """
)


