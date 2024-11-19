# missing_value
import pandas as pd
import streamlit as st
import arfs.preprocessing as arfspp
import arfs.feature_selection as arfsfs
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer

# this is a streamlit app to deal whith missing values with arfs and other methods,such as mice
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="数据预处理", page_icon="⚗️")
st.markdown("<h1 style='text-align: center; color: black;'>⚗️数据预处理✅</h1>", unsafe_allow_html=True)
st.expander("关于").write('''这是一个数据预处理的应用,包含了对数据进行编码、过滤等操作。以上操作都是可选操作，根据数据的分析目的确定，
                        比如用于制作图表和统计分析的数据不用整数编码，对于机器学习的数据需要进行整数编码，这里的缺失值插补方法要求是数字变量，
                        所以需要对数据进行整数编码。对于机器学习来说，数据预处理过程通常是整合到机器学习pipeline中，这样可以保证使用原始数据
                        就可以进行训练和预测等过程。
                        
                        ''')

st.header('1.转换编码器📟')

st.sidebar.header('关注作者')
st.sidebar.write('''
                 📰微信公众号：医研趣与美；\n 
                 📰CSDN账号：医学预测模型的开发与应用研究;\n 
                 🌐医学APP矩阵：app.clinicalmodelmatrix.com（需账号）
                 ''')
# load data
st.sidebar.divider()
upload_file = st.sidebar.file_uploader("上传数据", type=["csv"])

if upload_file is not None:
    df = pd.read_csv(upload_file)
    
#delete selected columns
st.sidebar.divider()
st.sidebar.header('0.删除选中的变量')
if upload_file is not None:
    delete_columns = st.sidebar.multiselect("select columns to delete", list(df.columns))
    if delete_columns:
        df = df.drop(delete_columns, axis=1) 
        st.sidebar.success("{} 列已经被删除".format(len(delete_columns)))   
# encoder
    #将指定的变量给成category类型
col1,col2=st.columns(2)
#将指定的变量给成category类型
with col1:
    st.write('1-1.类型转换-将指定的变量转成category类型')
    if upload_file is not None:
        var_category = st.multiselect("选择待转换类型的变量", list(df.columns))
        df[var_category] = df[var_category].astype("category")
        st.write(df.dtypes)
        st.download_button(
                label="Download data as CSV",
                data=df.to_csv(index=False,encoding='utf-8'),
                file_name='file.csv',
                mime='text/csv',
                key='download_category'
            )
with col2:
    #ordinal encoding
    st.write('1-2.整数编码-将指定的变量进行整数编码')
    if upload_file is not None:
        var_ordinal = st.multiselect("选择待整数编码的变量", list(df.columns))
        df[var_ordinal] = arfspp.OrdinalEncoder().fit_transform(df[var_ordinal])
        st.write(df.head(10))
        st.download_button(
                label="Download data as CSV",
                data=df.to_csv(index=False,encoding='utf-8'),
                file_name='file.csv',
                mime='text/csv',
                key='download_ordinal'
            )
st.divider()
st.header('2.过滤器⚗️')
#filter columns(variables)
col3,col4,col5=st.columns(3)
with col3:
    missing_threshold=st.slider("缺失值阈值%", 1, 100, 10)
    missing_filter=st.checkbox('2-1.缺失值过滤-过滤缺失值大于{}的变量'.format(missing_threshold),value=False)
    if upload_file is not None:
        if missing_filter:
            selector = arfsfs.MissingValueThreshold(threshold=missing_threshold/100)
            df = selector.fit_transform(df)
            st.write(df.head(10))
            st.download_button(
                label="Download data as CSV",
                data=df.to_csv(index=False,encoding='utf-8'),
                file_name='file.csv',
                mime='text/csv',
                key='download_missing'
            )
with col4:
    unique_threshold=st.slider("单调值阈值", 1, 5, 1)
    unique_filter=st.checkbox('2-2.单调值过滤-过滤单调值为{}的变量'.format(unique_threshold),value=False)
    
    if upload_file is not None:
        if unique_filter:
            selector = arfsfs.UniqueValuesThreshold(threshold=unique_threshold)
            df = selector.fit_transform(df)
            st.write(df.head(10))
            st.download_button(
                label="Download data as CSV",
                data=df.to_csv(index=False,encoding='utf-8'),
                file_name='file.csv',
                mime='text/csv',
                key='download_unique'
            )
        
with col5:
    cardinality_threshold=st.slider("相似度阈值%", 0, 100, 100)
    cardinality_filter=st.checkbox('2-3.相似值过滤-相似度值为{}%的变量'.format(cardinality_threshold),value=False)
    
    if upload_file is not None:
        if cardinality_filter:
            selector = arfsfs.CardinalityThreshold(threshold=100)
            df = selector.fit_transform(df)
            st.write(df.head(10))
            st.download_button(
                label="Download data as CSV",
                data=df.to_csv(index=False,encoding='utf-8'),
                file_name='file.csv',
                mime='text/csv',
                key='download_cardinality'
            )

st.divider()
st.header('3.缺失值填充🚰')
    
col6,col7=st.columns(2)
with col6:
    knn_imputer=st.checkbox('3-1.缺失值填充-knn imputer',value=False)
    if upload_file is not None:
        if knn_imputer:
            imputer = KNNImputer(n_neighbors=5)
            df_array=imputer.fit_transform(df)
            st.write(pd.DataFrame(df_array,columns=df.columns).head(10))
            st.download_button(
                label="Download data as CSV",
                data=pd.DataFrame(df_array,columns=df.columns).to_csv(index=False,encoding='utf-8'),
                file_name='file.csv',
                mime='text/csv',
                key='download_knn'
            )
            
with col7:
    mice_imputer=st.checkbox('3-2.缺失值填充-mice imputer',value=False)
    if upload_file is not None:
        if mice_imputer:
            imputer = IterativeImputer()
            df_array=imputer.fit_transform(df)
            st.write(pd.DataFrame(df_array,columns=df.columns).head(10))
        #download the df_array
            st.download_button(
                label="Download data as CSV",
                data=pd.DataFrame(df_array,columns=df.columns).to_csv(index=False,encoding='utf-8'),
                file_name='file.csv',
                mime='text/csv',
                key='download_mice'
            )
st.divider()
col8,col9,col10=st.columns([1,6,1])
with col9:
    st.warning("注：如果数据集中的变量名称是中文会显示乱码，请用excel的数据->来自文本->选择逗号分隔符来导入下载的文件。")