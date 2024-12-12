import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import random
import pickle
import surprise

# ĐỊNH NGHĨA CÁC HÀM:
# Random color:
def random_color():
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

# Hàm nhận gọi ý sản phẩm từ Content-Based:
def get_recommendations(df, ma_san_pham, cosine_sim, nums=5):
    # Get the index of the product that matches the ma_san_pham
    matching_indices = df.index[df['ma_san_pham'] == ma_san_pham].tolist()
    if not matching_indices:
        print(f"No product found with ID: {ma_san_pham}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the nums most similar products (Ignoring the product itself)
    sim_scores = sim_scores[1:nums+1]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top n most similar products as a DataFrame
    return df.iloc[product_indices]

# Hiển thị đề xuất ra bảng:
def display_recommended_products(recommended_products, cols=5):
    for i in range(0, len(recommended_products), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_products):
                product = recommended_products.iloc[i + j]
                with col:   
                    st.write(product['ten_san_pham'])                    
                    expander = st.expander(f"Mô tả")
                    product_description = product['mo_ta']
                    truncated_description = ' '.join(product_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

# Vẽ biểu đồ thống kê:
def thong_ke_sp(df, ID_sp):
    df_find = df[df['ma_san_pham'] == ID_sp]
    total = df_find['ma_san_pham'].count()
    count_positive = df_find['cam_xuc'].apply(lambda x: 'Positive' in x).sum()
    count_negative = df_find['cam_xuc'].apply(lambda x: 'Negative' in x).sum()
    count_neutral = df_find['cam_xuc'].apply(lambda x: 'Neutral' in x).sum()
    # Tạo dữ liệu thống kê
    df_plot = {"count": [total, count_positive, count_negative, count_neutral] }
    # Tạo DataFrame
    df_plot  = pd.DataFrame(df_plot , index=["total", "positive", "negative", "neutral"])

    # VẼ BIỂU ĐỒ:
    plt.figure(figsize=(8, 5))      # Kích thước biểu đồ
    bar_plot = sns.barplot(y=df_plot.index, x=df_plot["count"], palette="coolwarm", orient="h")
    # Hiển thị giá trị trên các cột
    for bar in bar_plot.patches:
        bar_width = bar.get_width()                 # Lấy chiều rộng của mỗi cột
        bar_plot.text(
            bar_width + 0.0,                        # Đặt nhãn ngay bên phải cột
            bar.get_y() + bar.get_height() / 2,     # Vị trí nhãn theo chiều y
            f'{int(bar_width)}',                    # Nội dung nhãn (giá trị)
            ha='left', va='center', fontsize=12, color='black'
                    )
    # Cấu hình biểu đồ
    plt.title(f"Thống kê đánh giá cho sản phẩn ID = {ID_sp}", fontsize=16)
    plt.xlabel("Số lượng đánh giá", fontsize=14)
    plt.ylabel("Phân loại", fontsize=14)
    plt.xticks(fontsize=12, rotation=45)            # Xoay nhãn trục x nếu cần
    plt.yticks(fontsize=12)

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)

# Hàm đề xuất sar phẩm:
def recommender_product(final_data, ID, algorithm):
    # Đề xuất sản phẩm cho khách hàng dựa trên model SVD:
    df_score = final_data[["ma_san_pham"]]
    df_score['Predict_sao'] = df_score['ma_san_pham'].apply(lambda x: algorithm.predict(ID, x).est) 
    df_score = df_score.sort_values(by=['Predict_sao'], ascending=False)
    df_score = df_score.drop_duplicates(subset='ma_san_pham')
    # In danh sách sản phẩm đề xuất:
    df_recomment = pd.merge(df_score, final_data, on='ma_san_pham', how='inner')
    df_recomment = df_recomment.drop_duplicates(subset='ma_san_pham')
    df_recomment = df_recomment[['ma_san_pham','ten_san_pham', 'mo_ta', 'so_sao', 'Predict_sao']]
    df_rec_print = df_recomment[['ma_san_pham','ten_san_pham', 'so_sao', 'Predict_sao']]
    
    return df_rec_print, df_recomment

# Download danh sách sản phẩm đề xuất:
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

# Hàm chọn 10 sản phảm bán nhiều nhất:
def best_sale(df):
    best_sp = df[['ma_san_pham']].value_counts().reset_index()
    best_sp.columns = ['ma_san_pham', 'count']
    best_sp = best_sp.sort_values('count', ascending=False)
    best_sp = best_sp.head(10)
    return best_sp

def the_best_product(df, ID):
    df_best_product = df[df['ma_san_pham'] == ID]
    df_best_product = df_best_product[['ma_san_pham', 'ten_san_pham', 'mo_ta', 'diem_trung_binh', 'nam', 'thang', 'ngay', 'cam_xuc']]
    df_best_product_id = df_best_product[['nam']].value_counts().reset_index()
    df_best_product_id.columns = ['nam', 'count_nam']
    df_best_product_id.sort_values('nam', ascending=True)
    return df_best_product_id

def top_10_product(df):
    # Tạo danh sách màu ngẫu nhiên
    colors = [random_color() for _ in range(len(df))]
    # Vẽ biểu đồ với màu sắc ngẫu nhiên
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x = df.iloc[:, 0], y = df.iloc[:, 1], palette=colors)
    # Thêm giá trị số trên mỗi cột
    for bar in bars.patches:
        yval = bar.get_height()                                     # Lấy chiều cao của cột
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5,       # Đặt giá trị hơi trên cột
                round(yval, 2), ha='center', va='bottom', fontsize=10)
    # Thêm nhãn và tiêu đề
    plt.xlabel("Mã sản phẩm", fontsize=14)
    plt.ylabel("Số lượng đơn hàng", fontsize=14)
    plt.title("Biểu đồ cột 10 sản phẩm bán tốt nhất", fontsize=16)
    plt.xticks(fontsize=12, rotation=45)                            # Xoay nhãn trục x nếu cần
    plt.yticks(fontsize=12)
    # Hiển thị biểu đồ
    plt.tight_layout()
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)

def plot_best_product_id(df, ID):
    # Tạo danh sách màu ngẫu nhiên
    colors = [random_color() for _ in range(len(df))]
    # Vẽ biểu đồ với màu sắc ngẫu nhiên
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x = df.iloc[:, 0], y = df.iloc[:, 1], palette=colors)
    # Thêm giá trị số trên mỗi cột
    for bar in bars.patches:
        yval = bar.get_height()                                     # Lấy chiều cao của cột
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 5,       # Đặt giá trị hơi trên cột
                round(yval, 2), ha='center', va='bottom', fontsize=10)
    # Thêm nhãn và tiêu đề
    plt.xlabel("Năm", fontsize=14)
    plt.ylabel("Số lượng đơn hàng", fontsize=14)
    plt.title(f"Biểu đồ cột thống kê của sản phẩm có ID = {ID}", fontsize=16)
    plt.xticks(fontsize=12, rotation=45)                            # Xoay nhãn trục x nếu cần
    plt.yticks(fontsize=12)
    # Hiển thị biểu đồ
    plt.tight_layout()
    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)

# ĐỌC DỮ LIỆU:
# Đọc dữ liệu sản phẩm 1200 dòng:
df_products = pd.read_csv('San_pham.csv')
# Đọc giữ liệu tổng hợp:
df_comments = pd.read_csv('Danh_gia_Processing.csv')
# Đọc giữ liệu tổng hợp:
df_data = pd.read_csv('data_100.csv')
# Lấy 10 sản phẩm
random_products = df_products.head(n=20)
# print(random_products)
st.session_state.random_products = random_products

# THIẾT KẾ SIDEBAR:
# Tạo menu
menu = ["Cho nhà Sản xuất", "Cho Khách hàng"]
choice = st.sidebar.selectbox("**Lựa chọn**", menu)

# Giáo viên hướng dẫn:
st.sidebar.write("""#### Giảng viên hướng dẫn:
                    Khuất Thùy Phương""")
# st.sidebar.write("**Khuất Thùy Phương**")

# Ngày bảo vệ:
st.sidebar.write("""#### Thời gian bảo vệ: 
                 16/12/2024""")

# Hiển thị thông tin thành viên trong sidebar
st.sidebar.write("#### Thành viên thực hiện:")

# Thành viên 1: Nguyễn Văn Thông
st.sidebar.image("Thongnv.jpg", use_container_width=True)
st.sidebar.write("**Nguyễn Văn Thông**")

# Thành viên 2: Vũ Trần Ngọc Lan
st.sidebar.image("Lan.jpg", use_container_width=True)
st.sidebar.write("**Vũ Trần Ngọc Lan**")

###### Giao diện Streamlit ######
st.image('Hasaki_Pic.jpg', use_container_width=True)

# TRẠNG THÁI CÁC LỰA CHỌN:
if "chon_sp" not in st.session_state:
    st.session_state.chon_sp = False

# LỰA CHỌN PHÂN TÍCH:
if choice == 'Cho Khách hàng':
    # Khởi tạo 02 button về 02 đối tượng Khách hàng:
    left, right = st.columns(2)
    def chon_san_pham():
        st.session_state.chon_sp = True
    
    def chon_ID_kh():
        st.session_state.chon_sp = False

    button_new = left.button("**LỰA CHỌN THEO SẢN PHẨM**", on_click = chon_san_pham)
    button_old = right.button("**KHÁCH HÀNG ĐÃ CÓ TÀI KHOẢN**", on_click = chon_ID_kh)
    
    # Khách hàng mới:
    if st.session_state.chon_sp:
        st.subheader("DANH MỤC SẢN PHẨM CỦA HASAKI & MỜI BẠN LỰA CHỌN")
        #LỤA CHỌN SẢN PHẨM: THEO COSIN-SIMILARITY
        with open('Group_11_Sun_Cosine_Sim.pkl', 'rb') as f:
            cosine_sim_new = pickle.load(f)
        # Kiểm tra xem 'selected_ma_san_pham' đã có trong session_state hay chưa
        if 'selected_ma_san_pham' not in st.session_state:
            # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID sản phẩm đầu tiên
            st.session_state.selected_ma_san_pham = None

        # Theo cách cho người dùng chọn sản phẩm từ dropdown
        # Tạo một tuple cho mỗi sản phẩm, trong đó phần tử đầu là tên và phần tử thứ hai là ID
        product_options = [(row['ten_san_pham'], row['ma_san_pham']) for index, row in st.session_state.random_products.iterrows()]
        st.session_state.random_products
        # Tạo một dropdown với options là các tuple này
        selected_product = st.selectbox(
                                        "MỜI BẠN CHỌN SẢN PHẨM:",
                                        options=product_options,
                                        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
                                        )
        # Display the selected product
        st.write("Sản phẩm bạn đã chọn:", selected_product)

        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_ma_san_pham = selected_product[1]


        # Hiển thị thống kê đánh giá của sản phẩm này:
        if st.session_state.selected_ma_san_pham:
            st.write("ma_san_pham: ", st.session_state.selected_ma_san_pham)
            ID_sp = selected_product[1]
            if st.button("Hiển thị thống kê"):
                thong_ke_sp(df_comments, ID_sp)

            # Hiển thị thông tin sản phẩm được chọn:
            selected_product = df_products[df_products['ma_san_pham'] == st.session_state.selected_ma_san_pham]
            if not selected_product.empty:
                # PHẦN THỐNG KÊ SẢN PHẨM:
                st.write('#### BẠN VỪA CHỌN:')
                st.write('### ', selected_product['ten_san_pham'].values[0])
                product_description = selected_product['mo_ta'].values[0]
                truncated_description = ' '.join(product_description.split()[:100])
                st.write('##### Mô tả chi tiết sản phẩm:')
                st.write(truncated_description, '...')

                # PHẦN ĐỀ XUẤT SẢN PHẨM:
                st.write('##### CÁC SẢN PHẨM TƯƠNG TỰ, MỜI BẠN THAM KHẢO:')
                recommendations = get_recommendations(df_products, st.session_state.selected_ma_san_pham, cosine_sim=cosine_sim_new, nums=3) 
                display_recommended_products(recommendations, cols=3)
            else:
                st.write(f"KHÔNG TÌM THẤY SẢN PHẨM VỚI MÃ ID LÀ: {st.session_state.selected_ma_san_pham}")
    
    # if button_old:
    if not st.session_state.chon_sp:
        st.subheader("PHÂN TÍCH & ĐỀ XUẤT SẢN PHẨM TƯƠNG TỰ CHO KHÁCH HÀNG CŨ")
        
        # KHÁCH HÀNG ĐĂNG NHẬP: BẰNG ID
        ID_number = st.text_input("Nhập ID của bạn:")
        if st.button("Đăng nhập"):
            st.session_state.customer_id = ID_number
        
        if "customer_id" in st.session_state:
            if st.session_state.customer_id:  # Kiểm tra nếu người dùng đã nhập mã khách hàng
                st.write(f"Bạn đã nhập ID = **{st.session_state.customer_id}**")
                # Tìm sản phẩm trong DataFrame theo ID
                customer_ID = df_data[df_data['ma_khach_hang'] == float(st.session_state.customer_id)]
                
                if not customer_ID.empty:
                    # Hiển thị thông tin sản phẩm khách hàng đã mua: 
                    df_print = pd.DataFrame([(row['ten_san_pham'], str(row['ma_san_pham']), row['so_sao']) for index, row in customer_ID.drop_duplicates().head(5).iterrows()], 
                                            columns=['ten_san_pham', 'ma_san_pham', 'so_sao'])
                    st.success("DANH MỤC SẢN PHẨM BẠN ĐÃ MUA: Hiển thị tối đa 05 Sản phẩm")
                    # df_print['ma_san_pham'] = df_print['ma_san_pham'].astype(int)
                    st.table(df_print)
                            
                    with open('Group_11_Sun_Surprise.pkl', 'rb') as f:
                        algorithm = pickle.load(f)
                    
                    reccoment_df, descript_df = recommender_product(df_data, float(st.session_state.customer_id),algorithm)
                    reccoment_df['ma_san_pham'] = reccoment_df['ma_san_pham'].astype(int)
                    st.success("SẢN PHẨM ĐƯỢC ĐỀ XUẤT CHO BẠN: Tối đa 05 Sản phẩm")
                    st.table(reccoment_df.head(5))
                    csv = convert_df(reccoment_df)

                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name="reccomment_df.csv",
                        mime="text/csv",
                                        )
                    # Cập nhật session_state dựa trên lựa chọn hiện tại
                    selected_product = st.selectbox(
                                        "XEM ĐÁNH GIÁ CỦA SẢN PHẨM:",
                                        options=[(row['ten_san_pham'], row['ma_san_pham']) for index, row in reccoment_df.drop_duplicates().head(5).iterrows()],
                                        format_func=lambda x: x[0]  # Hiển thị tên sản phẩm
                                        )     
                    st.write("Tên sản phẩm:", selected_product[0])
                        
                    st.session_state.selected_ma_san_pham = selected_product
                    ID_sp = selected_product[1]
                    # Hiển thị thống kê đánh giá của sản phẩm này:
                    if st.button("Hiển thị thống kê"):
                        thong_ke_sp(df_comments, ID_sp)
                    # Xem thông tin mô tả:
                    df_sp_descript = descript_df[descript_df['ma_san_pham'] == ID_sp].drop_duplicates(subset=['ma_san_pham'])
                    array_desript = df_sp_descript.to_numpy()[0, 2]
                    rec_expander = st.expander(f"Mô tả")
                    rec_truncated_description = ' '.join(array_desript.split()[:100]) + '...'
                    rec_expander.write(rec_truncated_description)
                    rec_expander.markdown("Nhấn vào mũi tên để đóng hộp text này.") 
                        
                else:
                    st.write(f"Không tìm thấy ID khách hàng = {ID_number}.")

                
            else:
                st.warning("ID bạn mới nhập không tồn tại. Mời bạn thử lại!")
 

elif choice == 'Cho nhà Sản xuất':
    st.subheader("NHỮNG THÔNG KÊ CƠ BẢN")

    # Số sản phẩm khác nhau:
    tk_sp_khac = st.expander(f"##### 1. Tổng số sản phẩm hiện có:")
    tk_sp_khac.image("Sanpham_Khacnhau.jpg", use_container_width=True)
    tk_sp_khac.markdown("Nhấn vào mũi tên để đóng hộp biểu đồ") 

    # Sản phẩm bán theo năm:
    tk_sp_nam = st.expander(f"##### 2. Tổng sản phẩm bán theo năm")
    tk_sp_nam.image("Sanpha_theo_nam.jpg", use_container_width=True)
    tk_sp_nam.markdown("Nhấn vào mũi tên để đóng hộp biểu đồ") 

    # Sản phẩm bán theo tháng:
    tk_sp_thang = st.expander(f"##### 3. Sản phẩm bán theo tháng")
    tk_sp_thang.image("Sanpham_theo_thang.jpg", use_container_width=True)
    tk_sp_thang.markdown("Nhấn vào mũi tên để đóng hộp biểu đồ") 

    # Tạo một expander trong Streamlit để chứa biểu đồ
    tk_sp_best_sale = st.expander(f"##### 4. Top 10 sản phẩm bán tốt nhất")
    df_best_sale = best_sale(df_comments)
    # Nút hiển thị:
    if tk_sp_best_sale.button("Hiển thị biểu đồ Top 10 Sản Phẩm Bán Chạy Nhất"):
        top_10_product(df_best_sale)  # Vẽ biểu đồ trong expander

    # Xem cụ thể thống kê sản phẩm theo từng năm:
    if 'top_san_pham' not in st.session_state:
        # Định nghĩa:
        st.session_state.top_san_pham = None
    # Chọn từ top 10 sản phẩm:
    selected_top_10_product = st.selectbox("Xem thông kê của sản phẩm:",
                                            options=[(row['ma_san_pham'], row['count']) for index, row in df_best_sale.iterrows()],
                                            format_func=lambda x: x[0]  # Hiển thị mã sản phẩm
                                            )     
    # st.write("Mã sản phẩm:", selected_top_10_product[0])
        
    st.session_state.top_san_pham = selected_top_10_product
    ID_sp_the_best = selected_top_10_product[0]
    # Hiển thị thống kê đánh giá của sản phẩm này:
    df_the_best_product = the_best_product(df_comments, ID_sp_the_best)
    if st.button(f"Xem thống kê của product ID = {ID_sp_the_best}"):
        plot_best_product_id(df_the_best_product, ID_sp_the_best)
    
    # Khách hàng đánh giá:
    tk_kh_danhgia = st.expander(f"##### 4. Thông kê phần đánh giá của khách hàng")
    tk_kh_danhgia.image("Khachhang_Danhgia.jpg", use_container_width=True)
    tk_kh_danhgia.markdown("Nhấn vào mũi tên để đóng hộp biểu đồ") 


# streamlit run Group_11_app.py

# group11-recommender-system

